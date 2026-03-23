"""Train the planner model on planner JSONL records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from learned.planner.data import PlannerJsonlDataset, PLANNER_ROOM_ID_TO_ASPECT
from learned.planner.model import PlannerTransformer, PlannerTransformerConfig


_CONTACT_GAP_SLACK = 0.02


def _aspect_lookup_tensor(device: torch.device) -> torch.Tensor:
    max_room_id = max(PLANNER_ROOM_ID_TO_ASPECT.keys(), default=0)
    lookup = torch.ones(max_room_id + 1, dtype=torch.float32, device=device)
    for room_id, aspect in PLANNER_ROOM_ID_TO_ASPECT.items():
        lookup[room_id] = float(aspect)
    return lookup


def _predicted_room_extents(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    room_type_ids = batch["room_type_ids"]
    area_pred = outputs["area_ratio"].clamp_min(1e-6)
    aspect_lookup = _aspect_lookup_tensor(room_type_ids.device)
    aspects = aspect_lookup[room_type_ids].clamp_min(0.4)
    widths = torch.sqrt(area_pred * aspects)
    heights = area_pred / widths.clamp_min(1e-6)
    return widths * 0.5, heights * 0.5


def _contact_and_overlap_loss(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    room_mask = batch["room_mask"].bool()
    pair_mask = room_mask.unsqueeze(1) & room_mask.unsqueeze(2)
    diagonal = torch.eye(pair_mask.size(-1), device=pair_mask.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~diagonal

    contact_targets = batch["contact_targets"] > 0.5
    contact_mask = pair_mask & contact_targets
    non_contact_mask = pair_mask & ~contact_targets

    half_widths, half_heights = _predicted_room_extents(batch, outputs)
    centroid = outputs["centroid"]
    dx = torch.abs(centroid[..., 0].unsqueeze(2) - centroid[..., 0].unsqueeze(1))
    dy = torch.abs(centroid[..., 1].unsqueeze(2) - centroid[..., 1].unsqueeze(1))

    target_dx = half_widths.unsqueeze(2) + half_widths.unsqueeze(1)
    target_dy = half_heights.unsqueeze(2) + half_heights.unsqueeze(1)
    gap_x = dx - target_dx
    gap_y = dy - target_dy

    if contact_mask.any():
        horizontal_cost = F.smooth_l1_loss(gap_x, torch.zeros_like(gap_x), reduction="none") + F.relu(gap_y - _CONTACT_GAP_SLACK)
        vertical_cost = F.smooth_l1_loss(gap_y, torch.zeros_like(gap_y), reduction="none") + F.relu(gap_x - _CONTACT_GAP_SLACK)
        contact_cost = torch.minimum(horizontal_cost, vertical_cost)
        contact_loss = contact_cost[contact_mask].mean()
    else:
        contact_loss = torch.zeros((), dtype=torch.float32, device=centroid.device)

    if non_contact_mask.any():
        overlap_penalty = F.relu(-gap_x) * F.relu(-gap_y)
        overlap_loss = overlap_penalty[non_contact_mask].mean()
    else:
        overlap_loss = torch.zeros((), dtype=torch.float32, device=centroid.device)

    return contact_loss, overlap_loss


def planner_loss(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    room_mask = batch["room_mask"].bool()
    centroid_pred = outputs["centroid"][room_mask]
    centroid_target = batch["centroid_targets"][room_mask]
    area_pred = outputs["area_ratio"][room_mask]
    area_target = batch["area_targets"][room_mask]

    centroid_loss = F.mse_loss(centroid_pred, centroid_target)
    area_loss = F.mse_loss(area_pred, area_target)

    adjacency_mask = room_mask.unsqueeze(1) & room_mask.unsqueeze(2)
    diagonal = torch.eye(adjacency_mask.size(-1), device=adjacency_mask.device, dtype=torch.bool).unsqueeze(0)
    adjacency_mask = adjacency_mask & ~diagonal
    adjacency_pred = outputs["adjacency_logits"][adjacency_mask]
    adjacency_target = batch["adjacency_targets"][adjacency_mask]
    adjacency_loss = F.binary_cross_entropy_with_logits(adjacency_pred, adjacency_target)

    contact_loss, overlap_loss = _contact_and_overlap_loss(batch, outputs)

    total = centroid_loss + 0.5 * area_loss + 0.75 * adjacency_loss + 0.4 * contact_loss + 0.2 * overlap_loss
    return {
        "total": total,
        "centroid": centroid_loss,
        "area": area_loss,
        "adjacency": adjacency_loss,
        "contact": contact_loss,
        "overlap": overlap_loss,
    }


def evaluate(model: PlannerTransformer, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "centroid": 0.0, "area": 0.0, "adjacency": 0.0, "contact": 0.0, "overlap": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                room_type_ids=batch["room_type_ids"],
                plot_type_ids=batch["plot_type_id"],
                room_mask=batch["room_mask"],
            )
            losses = planner_loss(batch, outputs)
            for key, value in losses.items():
                totals[key] += float(value.item())
            batches += 1

    batches = max(batches, 1)
    return {key: value / batches for key, value in totals.items()}


def _load_optional_json(path: str) -> Optional[Dict]:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _default_metrics_output(save_path: Path) -> Path:
    suffix = save_path.suffix or ".pt"
    return save_path.with_name(f"{save_path.stem}_training_report.json") if suffix else save_path.with_suffix(".training_report.json")


def build_training_report(
    *,
    args,
    device: str,
    train_record_count: int,
    val_record_count: int,
    best_epoch: int,
    best_val_loss: float,
    history: list[Dict[str, float]],
    checkpoint_path: str,
    dataset_summary: Optional[Dict] = None,
) -> Dict:
    return {
        "train_path": args.train,
        "val_path": args.val,
        "checkpoint_path": checkpoint_path,
        "device": device,
        "epochs_requested": int(args.epochs),
        "batch_size": int(args.batch_size),
        "optimizer": {
            "name": "AdamW",
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "model": {
            "max_rooms": int(args.max_rooms),
            "d_model": int(args.d_model),
            "n_heads": int(args.n_heads),
            "n_layers": int(args.n_layers),
            "d_ff": int(args.d_ff),
            "dropout": float(args.dropout),
        },
        "dataset": {
            "train_record_count": int(train_record_count),
            "val_record_count": int(val_record_count),
            "summary": dataset_summary,
        },
        "best_epoch": int(best_epoch),
        "best_val_loss": round(float(best_val_loss), 6),
        "history": history,
    }


def train(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = PlannerJsonlDataset(args.train, max_rooms=args.max_rooms)
    val_dataset = PlannerJsonlDataset(args.val, max_rooms=args.max_rooms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = PlannerTransformer(
        PlannerTransformerConfig(
            max_rooms=args.max_rooms,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
        )
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = math.inf
    best_epoch = 0
    history: list[Dict[str, float]] = []
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_summary = _load_optional_json(args.dataset_summary)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "centroid": 0.0, "area": 0.0, "adjacency": 0.0, "contact": 0.0, "overlap": 0.0}
        batches = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                room_type_ids=batch["room_type_ids"],
                plot_type_ids=batch["plot_type_id"],
                room_mask=batch["room_mask"],
            )
            losses = planner_loss(batch, outputs)
            losses["total"].backward()
            optimizer.step()

            for key, value in losses.items():
                running[key] += float(value.item())
            batches += 1

        train_metrics = {key: value / max(batches, 1) for key, value in running.items()}
        val_metrics = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train": {key: round(float(value), 6) for key, value in train_metrics.items()},
            "val": {key: round(float(value), 6) for key, value in val_metrics.items()},
        }
        history.append(epoch_record)

        print(
            f"[planner][epoch {epoch}] "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_centroid={val_metrics['centroid']:.4f} "
            f"val_area={val_metrics['area']:.4f} "
            f"val_adj={val_metrics['adjacency']:.4f} "
            f"val_contact={val_metrics['contact']:.4f} "
            f"val_overlap={val_metrics['overlap']:.4f}"
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save(
                {
                    "config": model.cfg.__dict__,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "loss_breakdown": val_metrics,
                    "train_record_count": len(train_dataset),
                    "val_record_count": len(val_dataset),
                },
                save_path,
            )

    report = build_training_report(
        args=args,
        device=device,
        train_record_count=len(train_dataset),
        val_record_count=len(val_dataset),
        best_epoch=best_epoch,
        best_val_loss=best_val,
        history=history,
        checkpoint_path=str(save_path),
        dataset_summary=dataset_summary,
    )
    metrics_output = Path(args.metrics_output) if args.metrics_output else _default_metrics_output(save_path)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[planner] wrote training report to {metrics_output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Train the room planner transformer")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--save", default="learned/planner/checkpoints/room_planner.pt")
    parser.add_argument("--metrics-output", default="")
    parser.add_argument("--dataset-summary", default="")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-rooms", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
