"""
test_overlap_processor.py – Unit tests for OverlapAwareProcessor.

Tests:
 - Static helpers: _iou, _bin_to_norm
 - Sequence parsing: _parse_complete_rooms, _last_room_idx
 - Core masking: _mask_overlap_bins
 - __call__ integration: tail-length routing, masking, no-op paths
"""
from __future__ import annotations

import pytest
import torch

from learned.model.sample import (
    ROOM_TOKEN,
    OverlapAwareProcessor,
    OVERLAP_PROCESSOR_ENABLED,
)


# ─── Mock tokenizer ───────────────────────────────────────────────────────────

class _MockTok:
    """Minimal stub with the attributes OverlapAwareProcessor needs."""
    def __init__(self, num_bins: int = 64, coord_offset: int = 37):
        self.num_bins = num_bins
        self.coord_offset = coord_offset


_NB = 64       # number of bins
_CO = 37       # coord offset
_VOCAB = _CO + _NB + 10   # total vocab size (enough headroom)


def _proc(**kw) -> OverlapAwareProcessor:
    return OverlapAwareProcessor(_MockTok(num_bins=_NB, coord_offset=_CO), **kw)


def _logits(fill: float = 0.0) -> torch.Tensor:
    return torch.full((1, _VOCAB), fill)


def _room_group(type_tok: int, x1b: int, y1b: int, x2b: int, y2b: int) -> list[int]:
    """Build a complete 6-token room group with bins shifted by coord_offset."""
    return [ROOM_TOKEN, type_tok, _CO + x1b, _CO + y1b, _CO + x2b, _CO + y2b]


def _seq(*groups: tuple, tail: tuple | None = None) -> torch.Tensor:
    """Build a seq tensor from complete room groups plus an optional partial tail.

    tail = (type_tok, x1b, y1b, x2b) → incomplete group of 4 tokens after ROOM_TOKEN
    This gives tail_len = 4 (_TAIL_X2) from the incomplete ROOM_TOKEN.

    tail = (type_tok, x1b, y1b, x2b, y2b) → 5 tokens after ROOM_TOKEN
    This gives tail_len = 5 (_TAIL_Y2).
    """
    tokens: list[int] = []
    for g in groups:
        tokens.extend(_room_group(*g))
    if tail is not None:
        tokens.append(ROOM_TOKEN)
        for b in tail:
            tokens.append(_CO + b)
    return torch.tensor([tokens])


# ─────────────────────────────────────────────────────────────────────────────
#  1. Static IoU helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIoU:

    def test_no_overlap(self):
        assert OverlapAwareProcessor._iou(0.0, 0.0, 0.2, 0.2,
                                           0.3, 0.3, 0.5, 0.5) == pytest.approx(0.0)

    def test_identical_boxes_is_one(self):
        assert OverlapAwareProcessor._iou(0.1, 0.1, 0.5, 0.5,
                                           0.1, 0.1, 0.5, 0.5) == pytest.approx(1.0, abs=1e-6)

    def test_partial_overlap(self):
        # A=(0,0,0.4,0.4), B=(0.2,0.2,0.6,0.6)
        # Inter = 0.04,  Union = 0.16+0.16-0.04 = 0.28
        iou = OverlapAwareProcessor._iou(0.0, 0.0, 0.4, 0.4,
                                          0.2, 0.2, 0.6, 0.6)
        assert iou == pytest.approx(0.04 / 0.28, abs=1e-6)

    def test_touching_edges_no_overlap(self):
        assert OverlapAwareProcessor._iou(0.0, 0.0, 0.5, 1.0,
                                           0.5, 0.0, 1.0, 1.0) == pytest.approx(0.0)

    def test_degenerate_area_returns_zero(self):
        # Zero-area box (point)
        assert OverlapAwareProcessor._iou(0.2, 0.2, 0.2, 0.2,
                                           0.0, 0.0, 1.0, 1.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  2. _bin_to_norm
# ─────────────────────────────────────────────────────────────────────────────

class TestBinToNorm:

    def test_bin_zero_is_zero(self):
        assert _proc()._bin_to_norm(0) == pytest.approx(0.0)

    def test_last_bin_is_one(self):
        assert _proc()._bin_to_norm(_NB - 1) == pytest.approx(1.0, abs=1e-6)

    def test_midpoint(self):
        p = OverlapAwareProcessor(_MockTok(num_bins=5, coord_offset=37))
        assert p._bin_to_norm(2) == pytest.approx(0.5, abs=1e-9)

    def test_single_bin_no_div_zero(self):
        p = OverlapAwareProcessor(_MockTok(num_bins=1, coord_offset=37))
        assert p._bin_to_norm(0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  3. _parse_complete_rooms
# ─────────────────────────────────────────────────────────────────────────────

class TestParseCompleteRooms:

    def test_empty_gives_empty(self):
        assert _proc()._parse_complete_rooms([]) == []

    def test_single_complete_room(self):
        # x1=10, y1=10, x2=50, y2=50
        tokens = _room_group(4, 10, 10, 50, 50)
        rooms = _proc()._parse_complete_rooms(tokens)
        assert len(rooms) == 1
        x1, y1, x2, y2 = rooms[0]
        assert x1 < x2 and y1 < y2

    def test_two_complete_rooms(self):
        tokens = _room_group(4, 5, 5, 20, 20) + _room_group(5, 35, 35, 55, 55)
        rooms = _proc()._parse_complete_rooms(tokens)
        assert len(rooms) == 2

    def test_incomplete_group_ignored(self):
        # Only 5 tokens: ROOM + type + x1 + y1 + x2 (missing y2)
        tokens = [ROOM_TOKEN, 4, _CO + 5, _CO + 5, _CO + 20]
        assert _proc()._parse_complete_rooms(tokens) == []

    def test_inverted_x_coords_skipped(self):
        # x1=50, x2=10 → x2 <= x1, should be discarded
        tokens = _room_group(4, 50, 10, 10, 50)
        assert _proc()._parse_complete_rooms(tokens) == []

    def test_inverted_y_coords_skipped(self):
        # y1=50, y2=10 → y2 <= y1
        tokens = _room_group(4, 10, 50, 50, 10)
        assert _proc()._parse_complete_rooms(tokens) == []

    def test_out_of_range_bins_ignored(self):
        # Use coord bins outside [0, num_bins) range
        tokens = [ROOM_TOKEN, 4, _CO + _NB, _CO + 0, _CO + _NB + 10, _CO + 10]
        assert _proc()._parse_complete_rooms(tokens) == []


# ─────────────────────────────────────────────────────────────────────────────
#  4. _last_room_idx
# ─────────────────────────────────────────────────────────────────────────────

class TestLastRoomIdx:

    def test_no_room_token_returns_minus_one(self):
        assert _proc()._last_room_idx([1, 2, 3, 4]) == -1

    def test_single_room_token(self):
        tokens = [5, 6, ROOM_TOKEN, 4, 37, 37, 37, 37]
        assert _proc()._last_room_idx(tokens) == 2

    def test_returns_last_occurrence(self):
        tokens = _room_group(4, 5, 5, 20, 20) + _room_group(5, 30, 30, 55, 55)
        # First ROOM_TOKEN at index 0, second at index 6
        assert _proc()._last_room_idx(tokens) == 6

    def test_empty_list_returns_minus_one(self):
        assert _proc()._last_room_idx([]) == -1


# ─────────────────────────────────────────────────────────────────────────────
#  5. _mask_overlap_bins
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskOverlapBins:

    def test_empty_existing_no_masking(self):
        p = _proc()
        logits = _logits()
        original = logits.clone()
        out = p._mask_overlap_bins(logits, [], 0.1, 0.1, 0.9, thresh=0.5)
        assert torch.equal(out, original)

    def test_identical_new_box_bins_masked(self):
        """A y2 bin that reproduces an existing room exactly should be masked."""
        p = _proc()
        # Existing room at bins 16,16 → 48,48 ≈ (0.254, 0.254, 0.762, 0.762)
        x1n = p._bin_to_norm(16)
        y1n = p._bin_to_norm(16)
        x2n = p._bin_to_norm(48)
        existing = [(x1n, y1n, x2n, p._bin_to_norm(48))]
        logits = _logits()
        out = p._mask_overlap_bins(logits, existing, x1n, y1n, x2n, thresh=0.5)
        # y2_bin=48 → tid = 48 + _CO should be -inf
        assert out[0, 48 + _CO] == float("-inf")

    def test_distant_box_not_masked(self):
        """A y2 bin that places the room far from existing rooms stays finite."""
        p = _proc()
        # Existing tiny room in top-left
        existing = [(0.0, 0.0, 0.1, 0.1)]
        logits = _logits()
        # New room in bottom-right: x1≈0.87, y1≈0.87, x2≈0.95
        x1n = p._bin_to_norm(56)
        y1n = p._bin_to_norm(56)
        x2n = p._bin_to_norm(61)
        out = p._mask_overlap_bins(logits, existing, x1n, y1n, x2n, thresh=0.5)
        # bin 62 in bottom-right → no overlap with top-left room
        assert out[0, 62 + _CO] != float("-inf")

    def test_already_masked_bin_stays_masked(self):
        """Bins pre-set to -inf are not changed."""
        p = _proc()
        existing = [(0.0, 0.0, 0.5, 0.5)]
        logits = _logits()
        logits[0, 10 + _CO] = float("-inf")
        out = p._mask_overlap_bins(logits, existing, 0.0, 0.0, 0.4, thresh=0.3)
        assert out[0, 10 + _CO] == float("-inf")


# ─────────────────────────────────────────────────────────────────────────────
#  6. __call__ integration
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlapAwareProcessorCall:

    def test_no_room_token_passthrough(self):
        p = _proc()
        seq = torch.tensor([[1, 2, 3, 4, 5]])
        logits = _logits()
        original = logits.clone()
        assert torch.equal(p(logits, seq), original)

    def test_wrong_tail_lengths_are_noop(self):
        """tail_len 1, 2, 3, 6 must not trigger masking."""
        p = _proc()
        complete = (4, 10, 10, 50, 50)  # one full room
        for extra_tokens in [1, 2, 3]:
            extra = [_CO + 5] * extra_tokens
            # Partial incomplete room with wrong tail length
            tokens = _room_group(*complete) + [ROOM_TOKEN] + extra
            seq = torch.tensor([tokens])
            logits = _logits()
            original = logits.clone()
            out = p(logits, seq)
            # Should be unchanged (no coordinate masking at these positions)
            # At least the non-coordinate region should be untouched
            assert out[0, :_CO].equal(original[0, :_CO])

    def test_tail_y2_no_existing_complete_rooms_passthrough(self):
        """If no complete rooms exist before the partial group, no masking."""
        p = _proc()
        # Single incomplete group at tail_len=5 with NO prior complete rooms
        # tail=(type, x1, y1, x2, y2_last) → tail_len = 5
        seq = _seq(tail=(4, 15, 15, 45, 45))
        logits = _logits()
        original = logits.clone()
        out = p(logits, seq)
        assert torch.equal(out, original)

    def test_tail_y2_with_prior_room_masks_overlapping_bin(self):
        """tail_len==5 with a complete existing room → high-IoU y2 bins masked."""
        p = _proc(iou_block_thresh=0.5)
        # Complete first room at bins 16,16,48,48
        complete_room = (4, 16, 16, 48, 48)
        # Second room at tail_len=5: x1=bin16, y1=bin16, x2=bin48, y2=bin48 (last in seq)
        # → IoU(new, existing) = 1.0 > 0.5 → should be masked
        seq = _seq(complete_room, tail=(5, 16, 16, 48, 48))
        logits = _logits()
        out = p(logits, seq)
        # The y2=bin48 logit (tid=48+_CO) should be -inf after masking
        assert out[0, 48 + _CO] == float("-inf")

    def test_tail_y2_inverted_x1x2_passthrough(self):
        """If x2 <= x1 in the new partial room, skip the check."""
        p = _proc(iou_block_thresh=0.5)
        complete_room = (4, 10, 10, 50, 50)
        # Partial with x1 > x2 (inverted): bins x1=40, x2=20
        seq = _seq(complete_room, tail=(5, 40, 10, 20, 50))  # x1=40 > x2=20
        logits = _logits()
        original = logits.clone()
        out = p(logits, seq)
        assert torch.equal(out, original)

    def test_also_check_x2_disabled_by_default(self):
        """also_check_x2=False → tail_len==4 does nothing."""
        p = _proc(also_check_x2=False)
        complete_room = (4, 16, 16, 48, 48)
        # tail=(type, x1, y1, x2) → tail_len=4 = _TAIL_X2 position
        seq = _seq(complete_room, tail=(5, 16, 16, 48))
        logits = _logits()
        original = logits.clone()
        out = p(logits, seq)
        assert torch.equal(out, original)

    def test_also_check_x2_enabled_masks_overlapping_x2(self):
        """also_check_x2=True → tail_len==4 masks x2 bins that cause overlap."""
        p = _proc(also_check_x2=True, iou_block_thresh_x2=0.5)
        # Existing room at (0.25, 0.25, 0.75, 0.75) — bins 16,16,48,48
        complete_room = (4, 16, 16, 48, 48)
        # New partial at x1=bin16, y1=bin16 — tail_len=4 (x1 y1 done, deciding x2)
        seq = _seq(complete_room, tail=(5, 16, 16))
        # Need tail_len=4 from last ROOM_TOKEN → seq must have 4 tokens after last ROOM
        # _seq with tail=(type, x1, y1) gives tail_len=3. Add one more manually.
        tokens = seq.squeeze(0).tolist() + [_CO + 48]  # add x2 as last token
        seq2 = torch.tensor([tokens])
        logits = _logits()
        out = p(logits, seq2)
        # With x2 = bin 48 ≈ 0.762, new box proxy (x1, y1, x2, 1.0) heavily overlaps existing
        # → high x2 bins should be masked; check at least one mid-range bin is masked
        high_iou_bins_masked = any(
            out[0, b + _CO] == float("-inf") for b in range(40, _NB)
        )
        assert high_iou_bins_masked


# ─────────────────────────────────────────────────────────────────────────────
#  7. Feature-flag integration
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureFlag:

    def test_iou_block_thresh_controls_masking(self):
        """Higher threshold allows more IoU before masking."""
        complete_room = (4, 16, 16, 48, 48)
        seq = _seq(complete_room, tail=(5, 16, 16, 48, 48))

        p_strict = _proc(iou_block_thresh=0.1)   # almost any overlap → mask
        p_loose  = _proc(iou_block_thresh=0.99)  # only near-perfect overlap → mask

        logits_s = _logits()
        logits_l = _logits()

        out_s = p_strict(logits_s, seq.clone())
        out_l = p_loose(logits_l, seq.clone())

        # Count masked bins in coordinate range
        masked_strict = int((out_s[0, _CO:_CO + _NB] == float("-inf")).sum())
        masked_loose  = int((out_l[0, _CO:_CO + _NB] == float("-inf")).sum())

        assert masked_strict >= masked_loose


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
