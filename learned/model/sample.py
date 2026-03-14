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
    device: str = "cpu",
) -> List[RoomBox]:
    """Sample a layout with spec-constrained decoding.

    Parameters
    ----------
    spec : dict
        Must contain ``"rooms"`` list with ``{"type": ...}`` entries.
        Room counts are inferred from the list (e.g., two ``"Bedroom"``
        entries → model must generate exactly two bedrooms).
    """
    room_types = [r["type"] for r in spec.get("rooms", [])]
    prompt = build_prompt(tokenizer, building_type, room_types).to(device)

    processor = SpecConstrainedProcessor(
        tokenizer, spec, max_rooms=max_rooms,
    )

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
        min_rooms=sum(processor.required.values()) if processor.required else 1,
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
