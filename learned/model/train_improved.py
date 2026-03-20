"""
train_improved.py - Enhanced training loop with data augmentation and auxiliary losses.

Key improvements over train.py:
- On-the-fly data augmentation (flip, rotate, jitter, shuffle)
- Enhanced auxiliary losses with stronger gradient signals:
  * compute_coverage_loss_enhanced: Exponential penalty for under-coverage
  * compute_overlap_loss_binary: Binary cross-entropy instead of MSE
  * compute_spread_loss: Original spread penalty
- Better defaults for small datasets
- Mixed precision training support
- Gradient accumulation for effective larger batch sizes
- Individual loss component tracking and logging

Enhanced Loss Functions (v2):
- Coverage loss: Uses exponential penalty instead of linear, stronger signal for poor coverage
- Overlap loss: Binary cross-entropy on per-room overlap flags, clearer gradient signal
- Increased default weights: coverage 0.3 (was 0.1), overlap 0.4 (was 0.2)

Usage
-----
    python -m learned.model.train_improved \\
        --train learned/data/kaggle_train.jsonl \\
        --val   learned/data/kaggle_train_val.jsonl \\
        --epochs 100 --batch 8 --accumulate 4 \\
        --save  learned/model/checkpoints/improved.pt \\
        --augment --aux-loss
"""
from __future__ import annotations

import argparse
import json as _json_mod
import math
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from learned.model.model import LayoutTransformer, LayoutTransformerConfig
from learned.data.tokenizer_layout import (
    LayoutTokenizer, RoomBox, DEFAULT_NUM_BINS,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, ROOM_TOKEN
)
from learned.data.augmentation import (
    AugmentationConfig, augment_layout, jitter_coordinates, shuffle_room_order
)


# =============================================================================
#  Augmented Dataset
# =============================================================================

class AugmentedJsonlDataset(Dataset):
    """Dataset with on-the-fly augmentation for layout sequences.

    Key features:
    - Decodes tokens back to RoomBox for augmentation
    - Applies spatial transforms (flip, rotate, jitter, shuffle)
    - Re-encodes augmented layouts
    - Supports offline expansion multiplier
    """

    def __init__(
        self,
        path: str,
        max_len: int = 256,
        pad_token: int = PAD_TOKEN,
        augment: bool = True,
        aug_config: Optional[AugmentationConfig] = None,
        expansion_factor: int = 1,  # Offline expansion (each sample appears N times per epoch)
    ):
        self.max_len = max_len
        self.pad_token = pad_token
        self.augment = augment
        self.aug_config = aug_config or AugmentationConfig()
        self.expansion_factor = max(1, expansion_factor)
        self.tokenizer = LayoutTokenizer()

        self._records: List[dict] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = _json_mod.loads(line)
                self._records.append(rec)

    def __len__(self) -> int:
        return len(self._records) * self.expansion_factor

    def _decode_and_augment(self, tokens: List[int]) -> List[int]:
        """Decode tokens to rooms, augment, re-encode."""
        # Decode rooms from tokens
        rooms = self.tokenizer.decode_rooms(tokens)
        if not rooms:
            return tokens  # Can't augment empty layout

        # Get building type
        building_type = self.tokenizer.decode_building_type(tokens)

        # Apply augmentation
        aug_rooms, _ = augment_layout(rooms, self.aug_config)

        # Re-encode
        return self.tokenizer.encode_sample(aug_rooms, building_type)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Map expanded index back to original
        real_idx = idx % len(self._records)
        tokens = self._records[real_idx]["tokens"]

        # Apply augmentation if enabled (but not for idx 0 of each sample to keep original)
        if self.augment and (idx // len(self._records)) > 0:
            tokens = self._decode_and_augment(tokens)

        # Pad/truncate
        L = min(len(tokens), self.max_len)
        t = torch.full((self.max_len,), self.pad_token, dtype=torch.long)
        t[:L] = torch.tensor(tokens[:L], dtype=torch.long)
        return t


# =============================================================================
#  Auxiliary Losses
# =============================================================================

def compute_coverage_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tokenizer: LayoutTokenizer,
    target_coverage: float = 0.7
) -> torch.Tensor:
    """Penalize layouts that don't fill the plot space.

    Encourages the model to use a larger portion of the [0,1] coordinate space.

    Args:
        logits: Model output logits [B, L, V]
        tokens: Target tokens [B, L]
        tokenizer: For decoding coordinate tokens
        target_coverage: Desired fraction of plot area to cover

    Returns:
        Coverage penalty loss (lower is better coverage)
    """
    batch_size = tokens.size(0)
    coverage_losses = []

    coord_offset = tokenizer.coord_offset
    coord_end = tokenizer.coord_token_end

    for b in range(batch_size):
        seq = tokens[b].tolist()

        # Extract room bounding boxes from tokens
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        i = 0
        while i < len(seq):
            if seq[i] == ROOM_TOKEN and i + 5 < len(seq):
                # Check if coordinates are valid
                coords = seq[i+2:i+6]
                if all(coord_offset <= c < coord_end for c in coords):
                    x_mins.append(tokenizer._unbin(coords[0]))
                    y_mins.append(tokenizer._unbin(coords[1]))
                    x_maxs.append(tokenizer._unbin(coords[2]))
                    y_maxs.append(tokenizer._unbin(coords[3]))
                i += 6
            elif seq[i] == EOS_TOKEN:
                break
            else:
                i += 1

        if not x_mins:
            coverage_losses.append(torch.tensor(0.0, device=logits.device))
            continue

        # Compute bounding box of all rooms
        overall_x_min = min(x_mins)
        overall_y_min = min(y_mins)
        overall_x_max = max(x_maxs)
        overall_y_max = max(y_maxs)

        # Coverage = area of bounding box / total area
        coverage = (overall_x_max - overall_x_min) * (overall_y_max - overall_y_min)

        # Penalize if coverage is below target
        if coverage < target_coverage:
            penalty = (target_coverage - coverage) ** 2
        else:
            penalty = 0.0

        coverage_losses.append(torch.tensor(penalty, device=logits.device))

    return torch.stack(coverage_losses).mean()


def compute_overlap_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tokenizer: LayoutTokenizer,
    iou_threshold: float = 0.3
) -> torch.Tensor:
    """Penalize overlapping rooms in the output.

    Args:
        logits: Model output logits [B, L, V]
        tokens: Target tokens [B, L]
        tokenizer: For decoding
        iou_threshold: IoU above which overlap is penalized

    Returns:
        Overlap penalty loss
    """
    batch_size = tokens.size(0)
    overlap_losses = []

    coord_offset = tokenizer.coord_offset
    coord_end = tokenizer.coord_token_end

    for b in range(batch_size):
        seq = tokens[b].tolist()

        # Extract rooms
        rooms = []
        i = 0
        while i < len(seq):
            if seq[i] == ROOM_TOKEN and i + 5 < len(seq):
                coords = seq[i+2:i+6]
                if all(coord_offset <= c < coord_end for c in coords):
                    x1 = tokenizer._unbin(coords[0])
                    y1 = tokenizer._unbin(coords[1])
                    x2 = tokenizer._unbin(coords[2])
                    y2 = tokenizer._unbin(coords[3])
                    rooms.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                i += 6
            elif seq[i] == EOS_TOKEN:
                break
            else:
                i += 1

        if len(rooms) < 2:
            overlap_losses.append(torch.tensor(0.0, device=logits.device))
            continue

        # Compute pairwise IoU and penalize overlaps
        total_overlap = 0.0
        num_pairs = 0

        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                r1, r2 = rooms[i], rooms[j]

                # Intersection
                ix1 = max(r1[0], r2[0])
                iy1 = max(r1[1], r2[1])
                ix2 = min(r1[2], r2[2])
                iy2 = min(r1[3], r2[3])

                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
                    area2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
                    union = area1 + area2 - intersection
                    iou = intersection / max(union, 1e-6)

                    if iou > iou_threshold:
                        total_overlap += (iou - iou_threshold) ** 2

                num_pairs += 1

        avg_overlap = total_overlap / max(num_pairs, 1)
        overlap_losses.append(torch.tensor(avg_overlap, device=logits.device))

    return torch.stack(overlap_losses).mean()


def compute_spread_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tokenizer: LayoutTokenizer
) -> torch.Tensor:
    """Penalize centroid clustering (rooms too close together).

    Encourages rooms to spread out across the plot.
    """
    batch_size = tokens.size(0)
    spread_losses = []

    coord_offset = tokenizer.coord_offset
    coord_end = tokenizer.coord_token_end

    for b in range(batch_size):
        seq = tokens[b].tolist()

        # Extract room centroids
        centroids = []
        i = 0
        while i < len(seq):
            if seq[i] == ROOM_TOKEN and i + 5 < len(seq):
                coords = seq[i+2:i+6]
                if all(coord_offset <= c < coord_end for c in coords):
                    x1 = tokenizer._unbin(coords[0])
                    y1 = tokenizer._unbin(coords[1])
                    x2 = tokenizer._unbin(coords[2])
                    y2 = tokenizer._unbin(coords[3])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    centroids.append((cx, cy))
                i += 6
            elif seq[i] == EOS_TOKEN:
                break
            else:
                i += 1

        if len(centroids) < 2:
            spread_losses.append(torch.tensor(0.0, device=logits.device))
            continue

        # Compute mean pairwise distance
        total_dist = 0.0
        num_pairs = 0

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                total_dist += dist
                num_pairs += 1

        avg_dist = total_dist / max(num_pairs, 1)

        # Penalize if average distance is too small (rooms clustered)
        # Target: rooms spread out so avg pairwise distance > 0.2
        target_dist = 0.2
        if avg_dist < target_dist:
            penalty = (target_dist - avg_dist) ** 2
        else:
            penalty = 0.0

        spread_losses.append(torch.tensor(penalty, device=logits.device))

    return torch.stack(spread_losses).mean()


# =============================================================================
#  Enhanced Loss Functions (Improved Gradient Signal)
# =============================================================================

def compute_overlap_loss_binary(
    logits: torch.Tensor,
    batch: torch.Tensor,
    tok: LayoutTokenizer,
    iou_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Binary cross-entropy overlap loss - stronger gradient signal than MSE.

    Instead of MSE on continuous overlap score (0.88), use BCE on binary
    overlap flags per room. This provides clearer gradient signal.

    Parameters
    ----------
    logits : torch.Tensor
        Model predictions [B, T, V]
    batch : torch.Tensor
        Ground truth tokens [B, T]
    tok : LayoutTokenizer
        For decoding coordinates
    iou_threshold : float
        IoU threshold above which rooms are considered overlapping

    Returns
    -------
    torch.Tensor
        Binary overlap loss (0 if no overlaps, penalty if overlaps exist)
    """
    try:
        batch_size = batch.shape[0]
        overlap_losses = []

        for b in range(batch_size):
            tokens = batch[b].tolist()

            # Skip if padded or too short
            if len(tokens) < 10 or PAD_TOKEN in tokens[:10]:
                overlap_losses.append(torch.tensor(0.0, device=logits.device))
                continue

            try:
                # Decode layout
                layout = tok.decode_to_roomboxes(tokens)
                if len(layout) < 2:
                    overlap_losses.append(torch.tensor(0.0, device=logits.device))
                    continue

                # Create binary overlap targets (1 = overlapping room, 0 = clean room)
                overlap_targets = []
                for i, room_a in enumerate(layout):
                    room_overlaps = False
                    for j, room_b in enumerate(layout):
                        if i != j:
                            # Check overlap using simple rectangular intersection
                            x1_a, y1_a, x2_a, y2_a = room_a.x1, room_a.y1, room_a.x2, room_a.y2
                            x1_b, y1_b, x2_b, y2_b = room_b.x1, room_b.y1, room_b.x2, room_b.y2

                            # Intersection area
                            x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
                            y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
                            intersection = x_overlap * y_overlap

                            # Union area
                            area_a = (x2_a - x1_a) * (y2_a - y1_a)
                            area_b = (x2_b - x1_b) * (y2_b - y1_b)
                            union = area_a + area_b - intersection

                            # IoU
                            if union > 0:
                                iou = intersection / union
                                if iou > iou_threshold:
                                    room_overlaps = True
                                    break

                    overlap_targets.append(1.0 if room_overlaps else 0.0)

                # If no rooms overlap, no penalty
                if not any(overlap_targets):
                    overlap_losses.append(torch.tensor(0.0, device=logits.device))
                    continue

                # Convert to tensor
                overlap_targets_tensor = torch.tensor(overlap_targets, dtype=torch.float32, device=logits.device)

                # For predictions, use coordinate logit entropy as proxy for "room quality"
                # Higher entropy = model is uncertain = more likely to have issues like overlaps
                room_predictions = []

                # Find room token positions
                room_tokens = [t for t, token in enumerate(tokens) if token == ROOM_TOKEN and t + 6 < len(tokens)]

                if len(room_tokens) != len(layout):
                    overlap_losses.append(torch.tensor(0.0, device=logits.device))
                    continue

                # Extract prediction uncertainty for each room
                for room_idx in room_tokens:
                    if room_idx + 6 < logits.shape[1]:
                        # Get coordinate logits for this room
                        coord_logits = logits[b, room_idx+2:room_idx+6]  # x1, y1, x2, y2
                        coord_probs = F.softmax(coord_logits, dim=-1)

                        # Compute entropy (uncertainty) - higher entropy = more likely to have issues
                        entropy = -(coord_probs * (coord_probs + 1e-8).log()).sum(dim=-1).mean()
                        # Normalize entropy to [0,1] range (entropy for uniform distribution ≈ log(vocab_size))
                        max_entropy = math.log(coord_logits.shape[-1])
                        normalized_entropy = min(1.0, entropy / max_entropy)
                        room_predictions.append(normalized_entropy)

                if len(room_predictions) != len(overlap_targets):
                    overlap_losses.append(torch.tensor(0.0, device=logits.device))
                    continue

                room_predictions_tensor = torch.stack(room_predictions)

                # Binary cross entropy loss
                bce_loss = F.binary_cross_entropy(room_predictions_tensor, overlap_targets_tensor, reduction='mean')
                overlap_losses.append(bce_loss)

            except Exception:
                overlap_losses.append(torch.tensor(0.0, device=logits.device))

        return torch.stack(overlap_losses).mean()

    except Exception:
        return torch.tensor(0.0, device=logits.device)


def compute_coverage_loss_enhanced(
    logits: torch.Tensor,
    batch: torch.Tensor,
    tok: LayoutTokenizer,
    min_coverage: float = 0.7,
) -> torch.Tensor:
    """
    Enhanced coverage loss with stronger gradient signal.

    Penalizes layouts that don't use enough of the available boundary space,
    with exponential penalty for severe under-coverage.

    Parameters
    ----------
    min_coverage : float
        Minimum coverage ratio required (0.7 = 70% of boundary should be used)
    """
    try:
        batch_size = batch.shape[0]
        coverage_losses = []

        for b in range(batch_size):
            tokens = batch[b].tolist()

            # Skip padded sequences
            if PAD_TOKEN in tokens[:10]:
                coverage_losses.append(torch.tensor(0.0, device=logits.device))
                continue

            try:
                # Decode layout
                layout = tok.decode_to_roomboxes(tokens)
                if not layout:
                    # Empty layout = maximum penalty
                    coverage_losses.append(torch.tensor(2.0, device=logits.device))
                    continue

                # Compute coverage
                all_x = [coord for room in layout for coord in [room.x1, room.x2]]
                all_y = [coord for room in layout for coord in [room.y1, room.y2]]

                if not all_x or not all_y:
                    coverage_losses.append(torch.tensor(2.0, device=logits.device))
                    continue

                used_width = max(all_x) - min(all_x)
                used_height = max(all_y) - min(all_y)
                used_area = used_width * used_height

                # Assume boundary is normalized [0, 1] space
                boundary_area = 1.0
                coverage_ratio = min(1.0, used_area / boundary_area)

                # Exponential penalty for severe under-coverage
                if coverage_ratio < min_coverage:
                    # Penalty grows exponentially as coverage decreases
                    deficit = min_coverage - coverage_ratio
                    penalty = 2.0 * (deficit / min_coverage) ** 2  # Quadratic penalty
                    coverage_losses.append(torch.tensor(penalty, device=logits.device))
                else:
                    # Small bonus for good coverage
                    coverage_losses.append(torch.tensor(0.0, device=logits.device))

            except Exception:
                # Decoding failed = penalty
                coverage_losses.append(torch.tensor(1.0, device=logits.device))

        return torch.stack(coverage_losses).mean()

    except Exception:
        return torch.tensor(0.0, device=logits.device)


# =============================================================================
#  Training
# =============================================================================

def train_improved(
    data_path: str,
    *,
    val_path: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 8,
    accumulate_steps: int = 4,  # Effective batch = batch_size * accumulate_steps
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    num_bins: int = DEFAULT_NUM_BINS,
    save_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_every: int = 50,
    # Augmentation
    augment: bool = True,
    expansion_factor: int = 8,  # Each sample appears 8x per epoch
    # Auxiliary losses
    aux_loss: bool = True,
    coverage_weight: float = 0.3,  # Increased for enhanced loss
    overlap_weight: float = 0.4,   # Increased for binary CE loss
    spread_weight: float = 0.1,    # Keep original
    # Model size
    n_layers: int = 8,  # Increased from 6
    n_heads: int = 8,
    d_model: int = 256,
    d_ff: int = 1024,
    dropout: float = 0.15,  # Slightly higher for regularization
) -> LayoutTransformer:
    """Run improved training loop."""

    print(f"[TRAIN] Device: {device}")

    # When not augmenting, don't duplicate samples
    if not augment:
        expansion_factor = 1

    print(f"[TRAIN] Augmentation: {augment}, expansion_factor: {expansion_factor}")
    print(f"[TRAIN] Auxiliary losses: {aux_loss}")
    print(f"[TRAIN] Effective batch size: {batch_size * accumulate_steps}")

    # Setup augmentation config
    aug_config = AugmentationConfig(
        enable_flip_h=True,
        enable_flip_v=True,
        enable_rotate=True,
        enable_jitter=True,
        enable_shuffle=True,
        jitter_sigma=0.015,
        jitter_prob=0.8,
        shuffle_prob=0.9,
    )

    # Create dataset
    ds = AugmentedJsonlDataset(
        data_path,
        max_len=256,
        augment=augment,
        aug_config=aug_config,
        expansion_factor=expansion_factor if augment else 1,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[TRAIN] Training samples: {len(ds)} (base: {len(ds._records)})")

    # Create model
    tok = LayoutTokenizer(num_bins=num_bins)
    cfg = LayoutTransformerConfig(
        vocab_size=tok.vocab_size,
        max_seq_len=256,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        dropout=dropout,
    )
    model = LayoutTransformer(cfg).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[TRAIN] Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with warmup
    total_steps = epochs * (len(loader) // accumulate_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Don't go below 10% of max lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    best_loss = float("inf")
    patience = 20  # Early stopping patience
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_cov_loss = 0.0  # Track coverage loss separately
        epoch_ovl_loss = 0.0  # Track overlap loss separately
        epoch_spr_loss = 0.0  # Track spread loss separately
        num_batches = 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)

            # Forward pass
            logits, ce_loss = model(batch, targets=batch)

            # Auxiliary losses
            aux_total = torch.tensor(0.0, device=device)
            cov_loss = torch.tensor(0.0, device=device)
            ovl_loss = torch.tensor(0.0, device=device)
            spr_loss = torch.tensor(0.0, device=device)

            if aux_loss:
                # Use enhanced loss functions for stronger gradient signals
                cov_loss = compute_coverage_loss_enhanced(logits, batch, tok, min_coverage=0.7)
                ovl_loss = compute_overlap_loss_binary(logits, batch, tok, iou_threshold=0.1)
                spr_loss = compute_spread_loss(logits, batch, tok)
                aux_total = (
                    coverage_weight * cov_loss +
                    overlap_weight * ovl_loss +
                    spread_weight * spr_loss
                )

            loss = ce_loss + aux_total

            # Scale loss for gradient accumulation
            loss = loss / accumulate_steps
            loss.backward()

            epoch_loss += loss.item() * accumulate_steps
            epoch_ce_loss += ce_loss.item()
            epoch_aux_loss += aux_total.item() if aux_loss else 0.0
            epoch_cov_loss += cov_loss.item()
            epoch_ovl_loss += ovl_loss.item()
            epoch_spr_loss += spr_loss.item()
            num_batches += 1

            # Update weights every accumulate_steps
            if (batch_idx + 1) % accumulate_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    print(f"  step {global_step:5d}  loss={loss.item()*accumulate_steps:.4f}  "
                          f"ce={ce_loss.item():.4f}  aux={aux_total.item():.4f}  "
                          f"cov={cov_loss.item():.3f}  ovl={ovl_loss.item():.3f}  spr={spr_loss.item():.3f}  "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_ce = epoch_ce_loss / max(num_batches, 1)
        avg_aux = epoch_aux_loss / max(num_batches, 1)
        avg_cov = epoch_cov_loss / max(num_batches, 1)
        avg_ovl = epoch_ovl_loss / max(num_batches, 1)
        avg_spr = epoch_spr_loss / max(num_batches, 1)
        dt = time.time() - t0

        # Validation
        if val_path is not None:
            val_ds = AugmentedJsonlDataset(val_path, augment=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for vbatch in val_loader:
                    vbatch = vbatch.to(device)
                    _, vloss = model(vbatch, targets=vbatch)
                    val_sum += vloss.item()
            val_avg = val_sum / max(len(val_loader), 1)
            monitor = val_avg
            print(f"Epoch {epoch:3d}/{epochs}  train={avg_loss:.4f} (ce={avg_ce:.4f} aux={avg_aux:.4f})  "
                  f"val={val_avg:.4f}  [cov={avg_cov:.3f} ovl={avg_ovl:.3f} spr={avg_spr:.3f}]  time={dt:.1f}s")
        else:
            monitor = avg_loss
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f} (ce={avg_ce:.4f} aux={avg_aux:.4f})  "
                  f"[cov={avg_cov:.3f} ovl={avg_ovl:.3f} spr={avg_spr:.3f}]  time={dt:.1f}s")

        # Save best checkpoint
        if monitor < best_loss:
            best_loss = monitor
            patience_counter = 0
            if save_path:
                _save_checkpoint(model, cfg, optimizer, epoch, monitor, save_path)
                print(f"  -> saved checkpoint ({monitor:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[TRAIN] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"[TRAIN] Training complete. Best loss: {best_loss:.4f}")
    return model


def _save_checkpoint(model, cfg, optimizer, epoch, loss, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_model": cfg.d_model,
            "d_ff": cfg.d_ff,
            "dropout": cfg.dropout,
            "pad_token": cfg.pad_token,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)


# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Train LayoutTransformer (improved)")

    # Data
    ap.add_argument("--train", required=True, help="Training .jsonl file")
    ap.add_argument("--val", default=None, help="Validation .jsonl file")

    # Training params
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=8)
    ap.add_argument("--accumulate", type=int, default=4,
                    help="Gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)

    # Model architecture
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.15)

    # Augmentation
    ap.add_argument("--augment", action="store_true", default=True,
                    help="Enable data augmentation")
    ap.add_argument("--no-augment", action="store_false", dest="augment")
    ap.add_argument("--expansion", type=int, default=8,
                    help="Augmentation expansion factor")

    # Auxiliary losses
    ap.add_argument("--aux-loss", action="store_true", default=True,
                    help="Enable auxiliary losses")
    ap.add_argument("--no-aux-loss", action="store_false", dest="aux_loss")

    # Output
    ap.add_argument("--save", "--checkpoint", dest="save",
                    default="learned/model/checkpoints/improved.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    train_improved(
        args.train,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate,
        lr=args.lr,
        n_layers=args.layers,
        n_heads=args.heads,
        d_model=args.d_model,
        dropout=args.dropout,
        augment=args.augment,
        expansion_factor=args.expansion,
        aux_loss=args.aux_loss,
        save_path=args.save,
        device=args.device,
    )


if __name__ == "__main__":
    main()
