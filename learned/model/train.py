"""
train.py - Minimal CPU/GPU training loop for the LayoutTransformer.

Usage
-----
    # From .pt tensor file:
    python -m learned.model.train \\
        --data learned/data/sequences.pt \\
        --epochs 10 --checkpoint learned/model/checkpoints/kaggle_test.pt

    # From JSONL files with validation:
    python -m learned.model.train \\
        --train learned/data/kaggle_train.jsonl \\
        --val   learned/data/kaggle_train_val.jsonl \\
        --epochs 10 --batch 16 --lr 3e-4 \\
        --save  learned/model/checkpoints/kaggle_test.pt

CLI aliases
-----------
    --train  = input JSONL for training (alternative to --data)
    --val    = input JSONL for validation (optional; used for best-checkpoint selection)
    --batch  = --batch-size
    --save   = --checkpoint
"""
from __future__ import annotations

import argparse
import json as _json_mod
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from learned.model.model import LayoutTransformer, LayoutTransformerConfig
from learned.data.tokenizer_layout import LayoutTokenizer, DEFAULT_NUM_BINS, PAD_TOKEN


# =============================================================================
#  Datasets
# =============================================================================

class SequenceDataset(Dataset):
    """Wraps a padded tensor dict produced by build_sequences (*.pt)."""

    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.sequences: torch.Tensor = data["sequences"]  # [N, L]
        self.lengths: torch.Tensor = data["lengths"]       # [N]

    def __len__(self):
        return self.sequences.size(0)

    def __getitem__(self, idx):
        return self.sequences[idx]


class JsonlDataset(Dataset):
    """Loads token sequences from a .jsonl file (one record per line).

    Each line: {"tokens": [...], ...}
    """

    def __init__(self, path: str, max_len: int = 256, pad_token: int = PAD_TOKEN):
        self.max_len   = max_len
        self.pad_token = pad_token
        self._records: list[list[int]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = _json_mod.loads(line)
                self._records.append(rec["tokens"])

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx) -> torch.Tensor:
        tokens = self._records[idx]
        L = min(len(tokens), self.max_len)
        t = torch.full((self.max_len,), self.pad_token, dtype=torch.long)
        t[:L] = torch.tensor(tokens[:L], dtype=torch.long)
        return t


def _make_dataset(path: str) -> Dataset:
    if Path(path).suffix.lower() == ".jsonl":
        return JsonlDataset(path, max_len=256)
    return SequenceDataset(path)


# =============================================================================
#  Training
# =============================================================================

def train(
    data_path: str,
    *,
    val_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    num_bins: int = DEFAULT_NUM_BINS,
    save_path: Optional[str] = None,
    device: str = "cpu",
    log_every: int = 50,
) -> LayoutTransformer:
    """Run the training loop and return the trained model."""
    ds     = _make_dataset(data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    tok     = LayoutTokenizer(num_bins=num_bins)
    max_seq = ds.sequences.size(1) if hasattr(ds, "sequences") else 256
    cfg     = LayoutTransformerConfig(
        vocab_size=tok.vocab_size,
        max_seq_len=max_seq,
    )
    model     = LayoutTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup then cosine decay
    total_steps = epochs * len(loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = batch.to(device)
            _, loss = model(batch, targets=batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % log_every == 0:
                print(f"  step {global_step:5d}  loss={loss.item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        avg = epoch_loss / max(len(loader), 1)
        dt  = time.time() - t0

        # Optional validation pass
        if val_path is not None:
            val_ds     = _make_dataset(val_path)
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
            print(f"Epoch {epoch}/{epochs}  train={avg:.4f}  val={val_avg:.4f}  "
                  f"time={dt:.1f}s")
        else:
            monitor = avg
            print(f"Epoch {epoch}/{epochs}  avg_loss={avg:.4f}  time={dt:.1f}s")

        if monitor < best_loss:
            best_loss = monitor
            if save_path:
                _save_checkpoint(model, cfg, optimizer, epoch, monitor, save_path)
                print(f"  -> saved checkpoint ({monitor:.4f})")

    return model


def _save_checkpoint(model, cfg, optimizer, epoch, loss, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "n_layers":    cfg.n_layers,
            "n_heads":     cfg.n_heads,
            "d_model":     cfg.d_model,
            "d_ff":        cfg.d_ff,
            "dropout":     cfg.dropout,
            "pad_token":   cfg.pad_token,
        },
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss":  loss,
    }, path)


# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Train LayoutTransformer")

    # Input data: --data (legacy .pt) or --train (.jsonl)
    data_src = ap.add_mutually_exclusive_group(required=True)
    data_src.add_argument("--data",  dest="data",
                          help="Path to sequences.pt (tensor format)")
    data_src.add_argument("--train", dest="data",
                          help="Path to training .jsonl file")

    ap.add_argument("--val", default=None,
                    help="Optional validation .jsonl for best-checkpoint selection")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", "--batch", dest="batch_size",
                    type=int, default=16,
                    help="Training batch size (default 16)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--checkpoint", "--save", dest="checkpoint",
                    default="learned/model/checkpoints/kaggle_test.pt",
                    help="Save path for best checkpoint")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    train(
        args.data,
        val_path=args.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
