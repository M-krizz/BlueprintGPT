import json
from pathlib import Path

from learned.planner.prepare_dataset import build_split_summary, split_records


INPUT_JSONL = Path("outputs/test_planner_split_input.jsonl")
TRAIN_JSONL = Path("outputs/test_planner_split_train.jsonl")
VAL_JSONL = Path("outputs/test_planner_split_val.jsonl")



def test_split_records_program_mode_preserves_multi_record_programs_in_validation():
    records = [
        {"plot_type": "Custom", "rooms": [{"name": "Bedroom_1", "type": "Bedroom"}, {"name": "Kitchen_1", "type": "Kitchen"}]},
        {"plot_type": "Custom", "rooms": [{"name": "Bedroom_1", "type": "Bedroom"}, {"name": "Kitchen_1", "type": "Kitchen"}]},
        {"plot_type": "Custom", "rooms": [{"name": "Bedroom_1", "type": "Bedroom"}, {"name": "Bathroom_1", "type": "Bathroom"}]},
        {"plot_type": "Custom", "rooms": [{"name": "Bedroom_1", "type": "Bedroom"}, {"name": "Bathroom_1", "type": "Bathroom"}]},
        {"plot_type": "Custom", "rooms": [{"name": "LivingRoom_1", "type": "LivingRoom"}]},
    ]
    INPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    INPUT_JSONL.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    train_count, val_count = split_records(
        str(INPUT_JSONL),
        str(TRAIN_JSONL),
        str(VAL_JSONL),
        val_ratio=0.4,
        seed=7,
        split_mode="program",
    )

    train_lines = [json.loads(line) for line in TRAIN_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    val_lines = [json.loads(line) for line in VAL_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert train_count == 3
    assert val_count == 2
    assert len(val_lines) == 2

    val_programs = {
        tuple(sorted(room["type"] for room in record["rooms"]))
        for record in val_lines
    }
    assert ("Bathroom", "Bedroom") in val_programs
    assert ("Bedroom", "Kitchen") in val_programs



def test_build_split_summary_reports_program_overlap_and_holdout_counts():
    summary = build_split_summary(str(TRAIN_JSONL), str(VAL_JSONL))

    assert summary["train"]["record_count"] == 3
    assert summary["val"]["record_count"] == 2
    assert summary["program_overlap_count"] == 2
    assert summary["validation_only_program_count"] == 0
