import json
from pathlib import Path
from types import SimpleNamespace

import torch

from learned.planner.data import PlannerJsonlDataset
from learned.planner.train import build_training_report, planner_loss


TEST_JSONL = Path("outputs/test_planner_training_records.jsonl")



def test_planner_dataset_emits_contact_targets():
    TEST_JSONL.parent.mkdir(parents=True, exist_ok=True)
    TEST_JSONL.write_text(
        json.dumps(
            {
                "plot_type": "Custom",
                "rooms": [
                    {"name": "LivingRoom_1", "type": "LivingRoom", "area_ratio": 0.18, "centroid": [0.3, 0.5]},
                    {"name": "Kitchen_1", "type": "Kitchen", "area_ratio": 0.14, "centroid": [0.55, 0.5]},
                ],
                "adjacency": [["LivingRoom_1", "Kitchen_1"]],
                "contact_pairs": [["LivingRoom_1", "Kitchen_1"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = PlannerJsonlDataset(str(TEST_JSONL), max_rooms=4)
    item = dataset[0]

    assert item["contact_targets"][0, 1].item() == 1.0
    assert item["contact_targets"][1, 0].item() == 1.0



def test_planner_loss_prefers_contact_friendly_predictions():
    batch = {
        "room_type_ids": torch.tensor([[4, 3, 0, 0]], dtype=torch.long),
        "plot_type_id": torch.tensor([4], dtype=torch.long),
        "room_mask": torch.tensor([[True, True, False, False]]),
        "centroid_targets": torch.tensor([[[0.32, 0.5], [0.55, 0.5], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        "area_targets": torch.tensor([[0.18, 0.14, 0.0, 0.0]], dtype=torch.float32),
        "adjacency_targets": torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "contact_targets": torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
    }

    near_outputs = {
        "centroid": torch.tensor([[[0.34, 0.5], [0.59, 0.5], [0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32),
        "area_ratio": torch.tensor([[0.18, 0.14, 0.01, 0.01]], dtype=torch.float32),
        "adjacency_logits": torch.tensor(
            [[[-10.0, 3.0, -10.0, -10.0], [3.0, -10.0, -10.0, -10.0], [-10.0, -10.0, -10.0, -10.0], [-10.0, -10.0, -10.0, -10.0]]],
            dtype=torch.float32,
        ),
    }
    far_outputs = {
        "centroid": torch.tensor([[[0.15, 0.2], [0.85, 0.8], [0.5, 0.5], [0.5, 0.5]]], dtype=torch.float32),
        "area_ratio": torch.tensor([[0.18, 0.14, 0.01, 0.01]], dtype=torch.float32),
        "adjacency_logits": torch.tensor(
            [[[-10.0, 3.0, -10.0, -10.0], [3.0, -10.0, -10.0, -10.0], [-10.0, -10.0, -10.0, -10.0], [-10.0, -10.0, -10.0, -10.0]]],
            dtype=torch.float32,
        ),
    }

    near_loss = planner_loss(batch, near_outputs)
    far_loss = planner_loss(batch, far_outputs)

    assert near_loss["contact"].item() < far_loss["contact"].item()
    assert near_loss["total"].item() < far_loss["total"].item()



def test_build_training_report_includes_dataset_and_history_metadata():
    args = SimpleNamespace(
        train="train.jsonl",
        val="val.jsonl",
        epochs=5,
        batch_size=8,
        lr=3e-4,
        weight_decay=1e-4,
        max_rooms=20,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        dropout=0.1,
    )

    report = build_training_report(
        args=args,
        device="cuda",
        train_record_count=120,
        val_record_count=30,
        best_epoch=3,
        best_val_loss=0.421337,
        history=[
            {"epoch": 1, "train": {"total": 0.8}, "val": {"total": 0.7}},
            {"epoch": 2, "train": {"total": 0.6}, "val": {"total": 0.5}},
        ],
        checkpoint_path="learned/planner/checkpoints/room_planner.pt",
        dataset_summary={"program_overlap_count": 4},
    )

    assert report["device"] == "cuda"
    assert report["dataset"]["train_record_count"] == 120
    assert report["dataset"]["summary"]["program_overlap_count"] == 4
    assert report["best_epoch"] == 3
    assert report["history"][1]["val"]["total"] == 0.5
