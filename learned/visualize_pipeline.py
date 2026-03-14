"""
visualize_pipeline.py – Rich visualizations for every stage of the
learned floor-plan pipeline.

Generates four outputs (saved to ``outputs/`` and shown interactively):

1. **Annotation overlay**  – room bounding boxes drawn on a blank canvas
   matching the original image size, colour-coded by room type.
2. **Dataset statistics**  – bar charts of room-type frequency, rooms-per-plan
   histogram, and plot-type distribution.
3. **Training loss curve** – train vs. val loss per epoch from the checkpoint's
   ``loss_curve.json``.
4. **Generated layout**    – sample a layout from the trained model and render
   room boxes as a professional floor-plan figure.

Usage
-----
    python -m learned.visualize_pipeline                    # all stages
    python -m learned.visualize_pipeline --only annotation  # just one
    python -m learned.visualize_pipeline --only stats
    python -m learned.visualize_pipeline --only loss
    python -m learned.visualize_pipeline --only generate
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless – save PNGs without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

_INTERACTIVE = matplotlib.get_backend().lower() not in ("agg",)


def _show():
    """Only call plt.show() when running with an interactive backend."""
    if _INTERACTIVE:
        _show()

# ── Colour palette for room types ─────────────────────────────────────────────
ROOM_PALETTE = {
    "Bedroom":        "#ef4444",
    "Bathroom":       "#3b82f6",
    "Kitchen":        "#f97316",
    "DrawingRoom":    "#22c55e",
    "Garage":         "#a16207",
    "Lounge":         "#eab308",
    "Lobby":          "#d946ef",
    "Passage":        "#84cc16",
    "Stairs":         "#14b8a6",
    "Lawn":           "#06d6a0",
    "OpenSpace":      "#06b6d4",
    "Staircase":      "#8b5cf6",
    "SideGarden":     "#facc15",
    "Dining":         "#f472b6",
    "DressingArea":   "#fb923c",
    "Store":          "#a855f7",
    "PrayerRoom":     "#6b7280",
    "ServantQuarter": "#4f46e5",
    "Backyard":       "#34d399",
    "Laundry":        "#c4b5fd",
    "LivingRoom":     "#10b981",
    "WC":             "#60a5fa",
    "DiningRoom":     "#ec4899",
    "Study":          "#0ea5e9",
    "Storage":        "#7c3aed",
    "Balcony":        "#fbbf24",
    "Unknown":        "#9ca3af",
}

OUTPUT_DIR = Path("outputs/visualizations")


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_color(room_type: str) -> str:
    return ROOM_PALETTE.get(room_type, "#9ca3af")


# ===========================================================================
# 1. Annotation overlay
# ===========================================================================

def visualize_annotation(annotation_path: str, save: bool = True):
    """Draw room polygons/bboxes from a single annotation JSON."""
    with open(annotation_path, "r") as f:
        data = json.load(f)

    w = data["image_width"]
    h = data["image_height"]
    rooms = data.get("rooms", [])
    plan_id = data.get("plan_id", Path(annotation_path).stem)
    plot_type = data.get("plot_type", "")

    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    # Draw rooms
    legend_handles = {}
    for room in rooms:
        rtype = room["room_type"]
        color = _get_color(rtype)
        polygon = room.get("polygon", [])
        bx, by, bw, bh = room["bbox"]

        if polygon and len(polygon) >= 3:
            poly = mpatches.Polygon(polygon, closed=True,
                                    facecolor=color, edgecolor="white",
                                    alpha=0.55, lw=1.2)
            ax.add_patch(poly)
        else:
            rect = mpatches.Rectangle((bx, by), bw, bh,
                                      facecolor=color, edgecolor="white",
                                      alpha=0.55, lw=1.2)
            ax.add_patch(rect)

        # Label
        cx = bx + bw / 2
        cy = by + bh / 2
        ax.text(cx, cy, rtype, color="white", fontsize=7,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.8, edgecolor="none"))

        if rtype not in legend_handles:
            legend_handles[rtype] = mpatches.Patch(facecolor=color, edgecolor="white",
                                                   label=rtype, alpha=0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # flip y for image coords
    ax.set_aspect("equal")
    ax.set_title(f"Annotation: {plan_id}  ({plot_type})", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=8)

    if legend_handles:
        leg = ax.legend(handles=list(legend_handles.values()), loc="upper right",
                        fontsize=7, framealpha=0.85, facecolor="#1e293b",
                        edgecolor="#475569", labelcolor="white")

    plt.tight_layout()
    if save:
        _ensure_output_dir()
        out = OUTPUT_DIR / f"annotation_{plan_id}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    _show()
    return fig


def visualize_annotations_grid(annotation_dir: str, n: int = 6, save: bool = True):
    """Show a grid of annotation overlays from the dataset."""
    ann_dir = Path(annotation_dir)
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        print("No annotations found.")
        return

    # Pick a spread: some from each plot type
    by_type: dict[str, list] = {}
    for f in files:
        name = f.stem
        pt = name.split("_")[0] if "_" in name else "Unknown"
        by_type.setdefault(pt, []).append(f)

    selected = []
    per = max(1, n // len(by_type))
    for pt, flist in sorted(by_type.items()):
        selected.extend(random.Random(42).sample(flist, min(per, len(flist))))
    selected = selected[:n]

    cols = min(3, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    fig.patch.set_facecolor("#1e293b")

    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (ax, fpath) in enumerate(zip(axes, selected)):
        with open(fpath) as f:
            data = json.load(f)
        w, h = data["image_width"], data["image_height"]
        plan_id = data.get("plan_id", fpath.stem)
        plot_type = data.get("plot_type", "")

        ax.set_facecolor("#0f172a")
        for room in data.get("rooms", []):
            rtype = room["room_type"]
            color = _get_color(rtype)
            polygon = room.get("polygon", [])
            bx, by, bw, bh = room["bbox"]
            if polygon and len(polygon) >= 3:
                poly = mpatches.Polygon(polygon, closed=True,
                                        facecolor=color, edgecolor="white",
                                        alpha=0.6, lw=0.8)
                ax.add_patch(poly)
            else:
                rect = mpatches.Rectangle((bx, by), bw, bh,
                                          facecolor=color, edgecolor="white",
                                          alpha=0.6, lw=0.8)
                ax.add_patch(rect)
            cx, cy = bx + bw / 2, by + bh / 2
            ax.text(cx, cy, rtype, color="white", fontsize=5,
                    ha="center", va="center", fontweight="bold")

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_title(f"{plan_id}\n({plot_type}, {len(data.get('rooms', []))} rooms)",
                     color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#64748b", labelsize=6)

    # Hide unused axes
    for ax in axes[len(selected):]:
        ax.set_visible(False)

    fig.suptitle("Extracted Annotations – Sample Grid", color="white",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        out = OUTPUT_DIR / "annotations_grid.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    _show()
    return fig


# ===========================================================================
# 2. Dataset statistics
# ===========================================================================

def visualize_dataset_stats(annotation_dir: str, save: bool = True):
    """Bar charts: room-type frequency, rooms-per-plan, plot-type distribution."""
    ann_dir = Path(annotation_dir)
    files = sorted(ann_dir.glob("*.json"))
    if not files:
        print("No annotations found.")
        return

    room_type_counts: Counter = Counter()
    rooms_per_plan: list[int] = []
    plot_types: Counter = Counter()
    area_by_type: dict[str, list[int]] = {}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        rooms = data.get("rooms", [])
        rooms_per_plan.append(len(rooms))
        plot_types[data.get("plot_type", "Unknown")] += 1
        for r in rooms:
            rt = r["room_type"]
            room_type_counts[rt] += 1
            area_by_type.setdefault(rt, []).append(r.get("area_px", 0))

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#1e293b")
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

    # — (a) Room type frequency bar chart —
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#0f172a")
    types_sorted = room_type_counts.most_common()
    labels = [t for t, _ in types_sorted]
    counts = [c for _, c in types_sorted]
    colors = [_get_color(t) for t in labels]
    bars = ax1.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.invert_yaxis()
    ax1.set_xlabel("Count", color="#cbd5e1", fontsize=10)
    ax1.set_title("Room Type Frequency (all plans)", color="white",
                  fontsize=12, fontweight="bold")
    ax1.tick_params(colors="#94a3b8", labelsize=8)
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 str(count), color="white", va="center", fontsize=7)

    # — (b) Plot type pie chart —
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#0f172a")
    pt_labels = list(plot_types.keys())
    pt_values = list(plot_types.values())
    pt_colors = ["#3b82f6", "#22c55e", "#f97316", "#a855f7"][:len(pt_labels)]
    wedges, texts, autotexts = ax2.pie(
        pt_values, labels=pt_labels, autopct="%1.0f%%",
        colors=pt_colors, textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 0.8},
        startangle=90,
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax2.set_title("Plot Type Distribution", color="white",
                  fontsize=12, fontweight="bold")

    # — (c) Rooms per plan histogram —
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#0f172a")
    ax3.hist(rooms_per_plan, bins=range(min(rooms_per_plan), max(rooms_per_plan) + 2),
             color="#3b82f6", edgecolor="white", alpha=0.8)
    ax3.set_xlabel("Rooms per plan", color="#cbd5e1", fontsize=10)
    ax3.set_ylabel("Count", color="#cbd5e1", fontsize=10)
    ax3.set_title("Rooms-per-Plan Distribution", color="white",
                  fontsize=12, fontweight="bold")
    ax3.tick_params(colors="#94a3b8", labelsize=8)
    ax3.axvline(np.mean(rooms_per_plan), color="#ef4444", ls="--", lw=1.5,
                label=f"Mean={np.mean(rooms_per_plan):.1f}")
    ax3.legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    # — (d) Top room types: avg area boxplot —
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor("#0f172a")
    top_types = [t for t, _ in room_type_counts.most_common(10)]
    box_data = [area_by_type.get(t, [0]) for t in top_types]
    bp = ax4.boxplot(box_data, tick_labels=top_types, patch_artist=True, vert=True,
                     medianprops=dict(color="white", lw=1.5),
                     whiskerprops=dict(color="#94a3b8"),
                     capprops=dict(color="#94a3b8"),
                     flierprops=dict(marker=".", markerfacecolor="#94a3b8", markersize=3))
    for patch, t in zip(bp["boxes"], top_types):
        patch.set_facecolor(_get_color(t))
        patch.set_alpha(0.7)
        patch.set_edgecolor("white")
    ax4.set_ylabel("Area (px)", color="#cbd5e1", fontsize=10)
    ax4.set_title("Room Area Distribution (top 10 types)", color="white",
                  fontsize=12, fontweight="bold")
    ax4.tick_params(colors="#94a3b8", labelsize=7)
    plt.setp(ax4.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle(f"Kaggle Floorplan Dataset – {len(files)} plans, "
                 f"{sum(rooms_per_plan)} rooms",
                 color="white", fontsize=14, fontweight="bold", y=1.02)

    if save:
        _ensure_output_dir()
        out = OUTPUT_DIR / "dataset_statistics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    _show()
    return fig


# ===========================================================================
# 3. Training loss curve
# ===========================================================================

def visualize_loss_curve(loss_json_path: str = "learned/model/checkpoints/loss_curve.json",
                         save: bool = True):
    """Plot train vs val loss from the JSON saved during training."""
    path = Path(loss_json_path)
    if not path.exists():
        print(f"Loss file not found: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    # Handle both formats: list-of-dicts or dict-of-lists
    if isinstance(data, list):
        epochs = [d.get("epoch", i + 1) for i, d in enumerate(data)]
        train_loss = [d["train_loss"] for d in data]
        val_loss = [d["val_loss"] for d in data if "val_loss" in d]
    else:
        epochs = list(range(1, len(data["train_loss"]) + 1))
        train_loss = data["train_loss"]
        val_loss = data.get("val_loss", [])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#0f172a")

    ax.plot(epochs, train_loss, "o-", color="#3b82f6", lw=2, markersize=5,
            label="Train Loss")
    if val_loss:
        ax.plot(epochs[:len(val_loss)], val_loss, "s-", color="#ef4444", lw=2,
                markersize=5, label="Val Loss")

    ax.set_xlabel("Epoch", color="#cbd5e1", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", color="#cbd5e1", fontsize=11)
    ax.set_title("LayoutTransformer Training Curve", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.grid(True, alpha=0.15, color="#64748b")
    ax.legend(fontsize=10, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    # Annotate best val
    if val_loss:
        best_idx = int(np.argmin(val_loss))
        ax.annotate(f"Best: {val_loss[best_idx]:.4f}",
                    xy=(best_idx + 1, val_loss[best_idx]),
                    xytext=(best_idx + 1 + 1, val_loss[best_idx] + 0.1),
                    color="#fbbf24", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2))

    plt.tight_layout()
    if save:
        _ensure_output_dir()
        out = OUTPUT_DIR / "loss_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    _show()
    return fig


# ===========================================================================
# 4. Generated layout visualization
# ===========================================================================

def visualize_generated_layout(
    checkpoint_path: str = "learned/model/checkpoints/kaggle_test.pt",
    building_type: str = "Residential",
    temperature: float = 0.8,
    n_layouts: int = 3,
    save: bool = True,
):
    """Load model, sample layouts, render professional floor-plan figures."""
    import torch
    from learned.model.model import LayoutTransformer, LayoutTransformerConfig
    from learned.data.tokenizer_layout import LayoutTokenizer, BOS_TOKEN, EOS_TOKEN

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = LayoutTransformerConfig(**ckpt["config"])
    model = LayoutTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tok = LayoutTokenizer(num_bins=cfg.vocab_size - cfg.vocab_size + 256)
    tok = LayoutTokenizer()  # use defaults

    cols = min(3, n_layouts)
    rows = math.ceil(n_layouts / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    fig.patch.set_facecolor("#1e293b")

    if n_layouts == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()

    for i in range(n_layouts):
        ax = axes_flat[i]
        ax.set_facecolor("#0f172a")

        # Build prompt & generate
        prompt_tokens = tok.encode_condition(building_type)
        prompt = torch.tensor([prompt_tokens], dtype=torch.long)
        output = model.generate(prompt, max_new_tokens=200, temperature=temperature,
                                top_p=0.95, eos_token=EOS_TOKEN)
        tokens = output.squeeze(0).tolist()
        rooms = tok.decode_rooms(tokens)

        if not rooms:
            ax.text(0.5, 0.5, "No rooms generated", color="white",
                    ha="center", va="center", fontsize=12, transform=ax.transAxes)
            continue

        # Clamp to [0, 1]
        for r in rooms:
            r.x_min = max(0, r.x_min)
            r.y_min = max(0, r.y_min)
            r.x_max = min(1, r.x_max)
            r.y_max = min(1, r.y_max)

        # Draw boundary
        boundary = mpatches.Rectangle((0, 0), 1, 1, fill=False,
                                      edgecolor="#64748b", lw=2, ls="--")
        ax.add_patch(boundary)

        # Draw rooms
        legend_handles = {}
        for room in rooms:
            color = _get_color(room.room_type)
            w = room.x_max - room.x_min
            h = room.y_max - room.y_min
            if w <= 0 or h <= 0:
                continue
            rect = mpatches.FancyBboxPatch(
                (room.x_min, room.y_min), w, h,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="white",
                alpha=0.65, lw=1.0,
            )
            ax.add_patch(rect)

            cx = room.x_min + w / 2
            cy = room.y_min + h / 2
            fontsize = max(5, min(8, int(min(w, h) * 30)))
            ax.text(cx, cy, room.room_type, color="white", fontsize=fontsize,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=color,
                              alpha=0.85, edgecolor="none"))

            if room.room_type not in legend_handles:
                legend_handles[room.room_type] = mpatches.Patch(
                    facecolor=color, edgecolor="white",
                    label=room.room_type, alpha=0.7)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"Generated Layout #{i+1}  ({len(rooms)} rooms)",
                     color="white", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#64748b", labelsize=7)

        if legend_handles:
            ax.legend(handles=list(legend_handles.values()), loc="upper right",
                      fontsize=6, framealpha=0.85, facecolor="#1e293b",
                      edgecolor="#475569", labelcolor="white")

    for ax in axes_flat[n_layouts:]:
        ax.set_visible(False)

    fig.suptitle(f"AI-Generated Floor Plans  (temp={temperature})",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        out = OUTPUT_DIR / "generated_layouts.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    _show()
    return fig


# ===========================================================================
# 5. JSONL Training data visualization (decoded token sequences)
# ===========================================================================

def visualize_training_samples(
    jsonl_path: str = "learned/data/kaggle_train.jsonl",
    n: int = 6,
    save: bool = True,
):
    """Decode training token sequences back to rooms and render them."""
    from learned.data.tokenizer_layout import LayoutTokenizer

    tok = LayoutTokenizer()
    path = Path(jsonl_path)
    if not path.exists():
        print(f"JSONL not found: {path}")
        return

    lines = path.read_text().strip().split("\n")
    sample_lines = random.Random(42).sample(lines, min(n, len(lines)))

    cols = min(3, len(sample_lines))
    rows = math.ceil(len(sample_lines) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    fig.patch.set_facecolor("#1e293b")

    if len(sample_lines) == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()

    for idx, line in enumerate(sample_lines):
        rec = json.loads(line)
        rooms = tok.decode_rooms(rec["tokens"])
        plan_id = rec.get("plan_id", f"sample_{idx}")

        ax = axes_flat[idx]
        ax.set_facecolor("#0f172a")

        boundary = mpatches.Rectangle((0, 0), 1, 1, fill=False,
                                      edgecolor="#475569", lw=2, ls="--")
        ax.add_patch(boundary)

        for room in rooms:
            color = _get_color(room.room_type)
            w = room.x_max - room.x_min
            h = room.y_max - room.y_min
            if w <= 0 or h <= 0:
                continue
            rect = mpatches.FancyBboxPatch(
                (room.x_min, room.y_min), w, h,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="white",
                alpha=0.6, lw=0.8,
            )
            ax.add_patch(rect)
            cx = room.x_min + w / 2
            cy = room.y_min + h / 2
            fontsize = max(5, min(7, int(min(w, h) * 25)))
            ax.text(cx, cy, room.room_type, color="white", fontsize=fontsize,
                    ha="center", va="center", fontweight="bold")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"{plan_id}  ({len(rooms)} rooms)",
                     color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#64748b", labelsize=6)

    for ax in axes_flat[len(sample_lines):]:
        ax.set_visible(False)

    fig.suptitle("Training Data – Decoded Token Sequences", color="white",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        _ensure_output_dir()
        out = OUTPUT_DIR / "training_samples.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out}")
    _show()
    return fig


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize every stage of the learned floorplan pipeline")
    parser.add_argument("--only", choices=["annotation", "stats", "loss", "generate", "training"],
                        default=None, help="Run only one visualization")
    parser.add_argument("--annotation-dir", default="learned/data/annotations")
    parser.add_argument("--annotation-file", default=None,
                        help="Single annotation JSON for detailed view")
    parser.add_argument("--loss-json", default="learned/model/checkpoints/loss_curve.json")
    parser.add_argument("--checkpoint", default="learned/model/checkpoints/kaggle_test.pt")
    parser.add_argument("--jsonl", default="learned/data/kaggle_train.jsonl")
    parser.add_argument("--n-layouts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    save = not args.no_save
    target = args.only

    print("=" * 60)
    print("  Learned Floorplan Pipeline – Visualization Suite")
    print("=" * 60)

    if target is None or target == "annotation":
        print("\n[1/5] Annotation overlays …")
        if args.annotation_file:
            visualize_annotation(args.annotation_file, save=save)
        else:
            visualize_annotations_grid(args.annotation_dir, n=6, save=save)

    if target is None or target == "stats":
        print("\n[2/5] Dataset statistics …")
        visualize_dataset_stats(args.annotation_dir, save=save)

    if target is None or target == "training":
        print("\n[3/5] Training data samples …")
        visualize_training_samples(args.jsonl, n=6, save=save)

    if target is None or target == "loss":
        print("\n[4/5] Training loss curve …")
        visualize_loss_curve(args.loss_json, save=save)

    if target is None or target == "generate":
        print("\n[5/5] Generated layouts …")
        visualize_generated_layout(
            checkpoint_path=args.checkpoint,
            temperature=args.temperature,
            n_layouts=args.n_layouts,
            save=save,
        )

    print("\n" + "=" * 60)
    print(f"  All outputs saved to:  {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
