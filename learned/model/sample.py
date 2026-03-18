"""
sample.py – Load a trained LayoutTransformer checkpoint and sample layouts.

Usage
-----
    python -m learned.model.sample \
        --checkpoint learned/model/checkpoints/kaggle_test.pt \
        --n 3 --temperature 0.8

Programmatic
------------
    model, tok = load_model("learned/model/checkpoints/kaggle_test.pt")
    rooms = sample_layout(model, tok, building_type="Residential", temperature=0.8)

Constrained sampling
--------------------
    rooms = constrained_sample_layout(
        model, tok,
        spec={"rooms": [{"type": "Bedroom"}, {"type": "Kitchen"}, ...]},
        temperature=0.8,
    )
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Optional

import torch

from learned.data.tokenizer_layout import (
    LayoutTokenizer,
    RoomBox,
    BOS_TOKEN,
    EOS_TOKEN,
    ROOM_TOKEN,
    ROOM_TYPES,
)
from learned.model.model import LayoutTransformer, LayoutTransformerConfig


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration: Overlap-Aware Logit Processor
# ═══════════════════════════════════════════════════════════════════════════════

# Master switch – default OFF for safe initial rollout; enable via env var
OVERLAP_PROCESSOR_ENABLED = os.getenv("LEARNED_OVERLAP_PROCESSOR_ENABLED", "false").lower() == "true"

# IoU above this threshold → mask the candidate y2 bin (very conservative default)
IOU_BLOCK_THRESH = float(os.getenv("OVERLAP_PROCESSOR_IOU_THRESH", "0.8"))

# Also check at x2 position (tail_len==3) with a wider threshold – experimental
ALSO_CHECK_X2 = os.getenv("OVERLAP_PROCESSOR_CHECK_X2", "false").lower() == "true"
IOU_BLOCK_THRESH_X2 = float(os.getenv("OVERLAP_PROCESSOR_IOU_THRESH_X2", "0.95"))


def _normalize_room_boxes(rooms: List[RoomBox]) -> List[RoomBox]:
    """Clamp coordinates to [0,1] and enforce x_min<=x_max, y_min<=y_max."""
    fixed: list[RoomBox] = []
    for r in rooms:
        x1 = max(0.0, min(1.0, r.x_min))
        y1 = max(0.0, min(1.0, r.y_min))
        x2 = max(0.0, min(1.0, r.x_max))
        y2 = max(0.0, min(1.0, r.y_max))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        fixed.append(RoomBox(r.room_type, x1, y1, x2, y2))
    return fixed


# ── Load ──────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[LayoutTransformer, LayoutTokenizer]:
    """Load model weights + config from a checkpoint and return (model, tokenizer)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = LayoutTransformerConfig(**ckpt["config"])
    model = LayoutTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    tok = LayoutTokenizer(num_bins=cfg.vocab_size - 37)  # 4 special + 27 room + 6 cond
    return model, tok


# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(
    tokenizer: LayoutTokenizer,
    building_type: str = "Residential",
    room_types: Optional[List[str]] = None,
) -> torch.Tensor:
    """Encode a conditioning prompt as a 1×T tensor."""
    tokens = tokenizer.encode_condition(building_type, room_types)
    return torch.tensor([tokens], dtype=torch.long)


# ── Sample ────────────────────────────────────────────────────────────────────

def sample_layout(
    model: LayoutTransformer,
    tokenizer: LayoutTokenizer,
    building_type: str = "Residential",
    room_types: Optional[List[str]] = None,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 0,
    max_new_tokens: int = 200,
    device: str = "cpu",
) -> List[RoomBox]:
    """Sample a single layout and return decoded rooms."""
    prompt = build_prompt(tokenizer, building_type, room_types).to(device)
    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token=EOS_TOKEN,
        enforce_structure=True,
        room_token=ROOM_TOKEN,
        type_token_start=tokenizer.type_token_start,
        type_token_end=tokenizer.type_token_end,
        coord_token_start=tokenizer.coord_offset,
        coord_token_end=tokenizer.coord_token_end,
    )
    tokens = output.squeeze(0).tolist()
    return tokenizer.decode_rooms(tokens)


# ── Spec-Constrained Sampling ────────────────────────────────────────────────

class SpecConstrainedProcessor:
    """Logit processor that enforces spec constraints during autoregressive generation.

    Constraints applied:
    1. **Room-type masking** – when the model is about to emit a room-type token
       (the token after ``<ROOM>``), only spec-allowed types are valid.
    2. **Room-count caps** – once a room type reaches its required count from
       the spec, it is masked so no extras are generated.
     3. **EOS gating/encouragement** – ``<EOS>`` is blocked until all required
         rooms are generated; once satisfied, ``<EOS>`` is strongly encouraged.
    4. **Max-rooms hard stop** – Force EOS once ``max_rooms`` rooms exist.
    """

    def __init__(
        self,
        tokenizer: LayoutTokenizer,
        spec: Dict,
        *,
        max_rooms: int = 20,
        eos_boost: float = 5.0,
    ):
        self.tok = tokenizer
        self.max_rooms = max_rooms
        self.eos_boost = eos_boost

        # Build required counts from spec  e.g. {"Bedroom": 2, "Kitchen": 1}
        self.required: Counter = Counter()
        for r in spec.get("rooms", []):
            self.required[r["type"]] += 1

        # Allowed type token ids
        self._allowed_type_ids = set()
        for rtype in self.required:
            tid = tokenizer._type2tok.get(rtype)
            if tid is not None:
                self._allowed_type_ids.add(tid)

        # All valid type token ids (for fallback when no spec filter needed)
        self._all_type_ids = set(tokenizer._type2tok.values())

    def __call__(self, logits: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Modify *logits* in-place and return them."""
        tokens = seq.squeeze(0).tolist()

        # Count rooms generated so far
        generated: Counter = Counter()
        i = 0
        while i < len(tokens):
            if tokens[i] == ROOM_TOKEN and i + 1 < len(tokens):
                type_tok = tokens[i + 1]
                rtype = self.tok._tok2type.get(type_tok)
                if rtype:
                    generated[rtype] += 1
                i += 6  # skip full room group
            else:
                i += 1

        total_rooms = sum(generated.values())

        # Hard stop: force EOS if we've reached max rooms
        if total_rooms >= self.max_rooms:
            logits.fill_(float("-inf"))
            logits[:, EOS_TOKEN] = 0.0
            return logits

        # Determine if next token is a room-type position
        # Room groups are: ROOM_TOKEN, type, x1, y1, x2, y2
        # If last token was ROOM_TOKEN → next is type position
        last_tok = tokens[-1] if tokens else -1
        is_type_position = (last_tok == ROOM_TOKEN)

        if is_type_position and self._allowed_type_ids:
            # Mask all type tokens that are NOT allowed or have reached their cap
            still_needed = set()
            for rtype, need in self.required.items():
                if generated.get(rtype, 0) < need:
                    tid = self.tok._type2tok.get(rtype)
                    if tid is not None:
                        still_needed.add(tid)

            if still_needed:
                # Mask all type tokens except those still needed
                for tid in self._all_type_ids:
                    if tid not in still_needed:
                        logits[:, tid] = float("-inf")
            else:
                # All requirements met at type position → allow any or push EOS
                # (model may end naturally or add bonus rooms)
                pass

        # EOS gating/encouragement based on required-room completion
        all_satisfied = all(
            generated.get(rt, 0) >= cnt
            for rt, cnt in self.required.items()
        )
        if not all_satisfied:
            logits[:, EOS_TOKEN] = float("-inf")
        elif total_rooms > 0:
            logits[:, EOS_TOKEN] += self.eos_boost

        return logits


# ── Overlap-Aware Logit Processor ─────────────────────────────────────────────

class OverlapAwareProcessor:
    """Logit processor that prevents generating boxes that heavily overlap existing ones.

    Strategy
    --------
    Token groups have the fixed structure::

        ROOM_TOKEN  type_tok  x1_bin  y1_bin  x2_bin  y2_bin

    ``tail_len`` = number of tokens emitted **after** the last ROOM_TOKEN.

    * **tail_len == 5** (about to sample ``y2_bin``): we already know
      ``x1, y1, x2`` for the current room.  For every candidate ``y2_bin``
      token we compute the IoU of the resulting box against all fully-placed
      rooms.  Bins whose IoU > ``iou_block_thresh`` are masked.

    * **tail_len == 4** (about to sample ``x2_bin``, optional): check a
      proxy box ``(x1, y1, cand_x2, 1.0)`` – conservative worst-case height.
      Uses a stricter threshold; disabled by default.

    Design choices
    --------------
    * Conservative default (IoU > 0.8) avoids destroying diversity.
    * Skips check when no complete room exists yet.
    * Falls back silently on any parsing error.
    * Feature-flagged: ``LEARNED_OVERLAP_PROCESSOR_ENABLED=true`` to activate.
    """

    _TAIL_Y2 = 5   # tail length when about to sample y2_bin
    _TAIL_X2 = 4   # tail length when about to sample x2_bin (optional)

    def __init__(
        self,
        tokenizer: LayoutTokenizer,
        *,
        iou_block_thresh: float = IOU_BLOCK_THRESH,
        also_check_x2: bool = ALSO_CHECK_X2,
        iou_block_thresh_x2: float = IOU_BLOCK_THRESH_X2,
    ):
        self.tok = tokenizer
        self.iou_block_thresh = iou_block_thresh
        self.also_check_x2 = also_check_x2
        self.iou_block_thresh_x2 = iou_block_thresh_x2
        self.coord_offset = tokenizer.coord_offset
        self.num_bins = tokenizer.num_bins

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _bin_to_norm(self, b: int) -> float:
        return b / max(self.num_bins - 1, 1)

    @staticmethod
    def _iou(x1a: float, y1a: float, x2a: float, y2a: float,
              x1b: float, y1b: float, x2b: float, y2b: float) -> float:
        ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
        ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
        area_b = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
        return inter / max(area_a + area_b - inter, 1e-9)

    def _parse_complete_rooms(self, tokens: List[int]) -> List[tuple]:
        """Return (x1,y1,x2,y2) tuples for every complete room group in *tokens*."""
        rooms = []
        i = 0
        co = self.coord_offset
        nb = self.num_bins
        while i < len(tokens):
            if tokens[i] == ROOM_TOKEN and i + 5 < len(tokens):
                raw = [tokens[i + k] - co for k in range(2, 6)]
                if all(0 <= b < nb for b in raw):
                    x1, y1, x2, y2 = (self._bin_to_norm(b) for b in raw)
                    if x2 > x1 and y2 > y1:
                        rooms.append((x1, y1, x2, y2))
                i += 6
            else:
                i += 1
        return rooms

    def _last_room_idx(self, tokens: List[int]) -> int:
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == ROOM_TOKEN:
                return i
        return -1

    def _mask_overlap_bins(
        self,
        logits: torch.Tensor,
        existing: List[tuple],
        x1: float, y1: float,
        x2: Optional[float],
        thresh: float,
    ) -> torch.Tensor:
        """Mask coordinate bins that produce IoU > thresh with any existing room.

        *x2=None* → x2 position (proxy box with y2=1.0);
        *x2 given* → y2 position (full box check).
        """
        co = self.coord_offset
        nb = self.num_bins
        vcab = logits.shape[-1]

        for cand_bin in range(nb):
            tid = cand_bin + co
            if tid >= vcab:
                break
            if logits[0, tid] == float("-inf"):
                continue

            v = self._bin_to_norm(cand_bin)

            if x2 is None:           # x2 position
                cx2, cy2 = v, 1.0
                if cx2 <= x1:
                    continue
            else:                     # y2 position
                cx2, cy2 = x2, v
                if cy2 <= y1:
                    continue

            for (rx1, ry1, rx2, ry2) in existing:
                if self._iou(x1, y1, cx2, cy2, rx1, ry1, rx2, ry2) > thresh:
                    logits[0, tid] = float("-inf")
                    break

        return logits

    # ── Main interface ────────────────────────────────────────────────────────

    def __call__(self, logits: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        tokens = seq.squeeze(0).tolist()
        room_idx = self._last_room_idx(tokens)
        if room_idx < 0:
            return logits

        tail_len = len(tokens) - room_idx - 1

        # ── y2 position ───────────────────────────────────────────────────
        if tail_len == self._TAIL_Y2:
            try:
                co = self.coord_offset
                x1 = self._bin_to_norm(tokens[room_idx + 2] - co)
                y1 = self._bin_to_norm(tokens[room_idx + 3] - co)
                x2 = self._bin_to_norm(tokens[room_idx + 4] - co)
            except IndexError:
                return logits
            if x2 <= x1:
                return logits
            existing = self._parse_complete_rooms(tokens[:room_idx])
            if not existing:
                return logits
            return self._mask_overlap_bins(logits, existing, x1, y1, x2, self.iou_block_thresh)

        # ── x2 position (optional, conservative) ─────────────────────────
        if self.also_check_x2 and tail_len == self._TAIL_X2:
            try:
                co = self.coord_offset
                x1 = self._bin_to_norm(tokens[room_idx + 2] - co)
                y1 = self._bin_to_norm(tokens[room_idx + 3] - co)
            except IndexError:
                return logits
            existing = self._parse_complete_rooms(tokens[:room_idx])
            if not existing:
                return logits
            return self._mask_overlap_bins(logits, existing, x1, y1, None, self.iou_block_thresh_x2)

        return logits


# ── Spec-Constrained Sampling ────────────────────────────────────────────────

def constrained_sample_layout(
    model: LayoutTransformer,
    tokenizer: LayoutTokenizer,
    spec: Dict,
    *,
    building_type: str = "Residential",
    temperature: float = 0.85,
    top_p: float = 0.95,
    top_k: int = 0,
    max_new_tokens: int = 200,
    max_rooms: int = 20,
    min_x_bin_gap: int = 1,
    min_y_bin_gap: int = 1,
    overlap_processor: bool = OVERLAP_PROCESSOR_ENABLED,
    device: str = "cpu",
) -> List[RoomBox]:
    """Sample a layout with spec-constrained decoding.

    Parameters
    ----------
    spec : dict
        Must contain ``"rooms"`` list with ``{"type": ...}`` entries.
        Room counts are inferred from the list (e.g., two ``"Bedroom"``
        entries → model must generate exactly two bedrooms).
    overlap_processor : bool
        When True (and ``LEARNED_OVERLAP_PROCESSOR_ENABLED=true``), chain
        :class:`OverlapAwareProcessor` after the spec constraints to mask
        coordinate bins that would produce IoU > ``IOU_BLOCK_THRESH`` with
        already-placed rooms.
    """
    room_types = [r["type"] for r in spec.get("rooms", [])]
    prompt = build_prompt(tokenizer, building_type, room_types).to(device)

    spec_proc = SpecConstrainedProcessor(
        tokenizer, spec, max_rooms=max_rooms,
    )
    if overlap_processor:
        overlap_proc = OverlapAwareProcessor(tokenizer)
        def _chained(logits: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
            logits = spec_proc(logits, seq)
            logits = overlap_proc(logits, seq)
            return logits
        processor = _chained
    else:
        processor = spec_proc

    output = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token=EOS_TOKEN,
        logit_processor=processor,
        enforce_structure=True,
        room_token=ROOM_TOKEN,
        type_token_start=tokenizer.type_token_start,
        type_token_end=tokenizer.type_token_end,
        coord_token_start=tokenizer.coord_offset,
        coord_token_end=tokenizer.coord_token_end,
        min_rooms=sum(spec_proc.required.values()) if spec_proc.required else 1,
        min_x_bin_gap=min_x_bin_gap,
        min_y_bin_gap=min_y_bin_gap,
    )
    tokens = output.squeeze(0).tolist()
    decoded = tokenizer.decode_rooms(tokens)

    # Final cleanup: keep only requested room types and cap counts.
    required: Counter = Counter(r["type"] for r in spec.get("rooms", []))
    if not required:
        return decoded

    kept: list[RoomBox] = []
    used: Counter = Counter()
    for room in decoded:
        rtype = room.room_type
        if rtype not in required:
            continue
        if used[rtype] >= required[rtype]:
            continue
        kept.append(room)
        used[rtype] += 1

    return kept


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Sample layouts from a trained LayoutTransformer")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=3, help="Number of layouts to sample")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--building-type", default="Residential")
    ap.add_argument("--spec-file", default=None,
                    help="Optional JSON file with {'rooms':[{'type':'Bedroom'}, ...]} for constrained decoding")
    ap.add_argument("--constrained", action="store_true",
                    help="Enable constrained decoding using --spec-file")
    ap.add_argument("--normalize", action="store_true",
                    help="Clamp sampled coordinates into [0,1] and fix inverted boxes")
    ap.add_argument("--min-x-bin-gap", type=int, default=1,
                    help="Minimum x2-x1 gap in coordinate bins for constrained sampling")
    ap.add_argument("--min-y-bin-gap", type=int, default=1,
                    help="Minimum y2-y1 gap in coordinate bins for constrained sampling")
    ap.add_argument("--overlap-processor", action="store_true",
                    default=OVERLAP_PROCESSOR_ENABLED,
                    help="Enable overlap-aware logit processor during constrained decoding")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, tok = load_model(args.checkpoint, device=args.device)

    spec = None
    if args.constrained:
        if not args.spec_file:
            raise SystemExit("--constrained requires --spec-file")
        with open(args.spec_file, "r", encoding="utf-8") as fh:
            spec = json.load(fh)

    for i in range(args.n):
        if args.constrained and spec is not None:
            rooms = constrained_sample_layout(
                model, tok,
                spec=spec,
                building_type=args.building_type,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_x_bin_gap=args.min_x_bin_gap,
                min_y_bin_gap=args.min_y_bin_gap,
                overlap_processor=args.overlap_processor,
                device=args.device,
            )
        else:
            rooms = sample_layout(
                model, tok,
                building_type=args.building_type,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=args.device,
            )
        if args.normalize:
            rooms = _normalize_room_boxes(rooms)
        print(f"\nLayout {i+1}: {len(rooms)} rooms")
        for r in rooms:
            print(f"  {r.room_type:20s}  ({r.x_min:.3f},{r.y_min:.3f})->({r.x_max:.3f},{r.y_max:.3f})")


if __name__ == "__main__":
    main()
