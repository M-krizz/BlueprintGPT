"""
augmentation.py - Data augmentation for floor plan layouts.

Implements spatial transformations to expand effective training data:
- Horizontal/vertical flipping
- 90-degree rotations
- Coordinate jitter (small random perturbations)
- Room order shuffling (breaks autoregressive order bias)
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional

from learned.data.tokenizer_layout import RoomBox


@dataclass
class AugmentationConfig:
    """Configuration for layout augmentation."""
    enable_flip_h: bool = True       # Horizontal flip (left-right)
    enable_flip_v: bool = True       # Vertical flip (top-bottom)
    enable_rotate: bool = True       # 90/180/270 degree rotations
    enable_jitter: bool = True       # Small coordinate perturbations
    enable_shuffle: bool = True      # Room order shuffling

    jitter_sigma: float = 0.01       # Std dev for coordinate jitter (normalized)
    jitter_clamp: float = 0.03       # Max jitter magnitude

    # Probability of applying each augmentation
    flip_prob: float = 0.5
    rotate_prob: float = 0.5
    jitter_prob: float = 0.7
    shuffle_prob: float = 0.8


def flip_horizontal(rooms: List[RoomBox]) -> List[RoomBox]:
    """Flip layout horizontally (mirror across Y axis).

    x' = 1 - x, coordinates swap: x_min <-> x_max
    """
    result = []
    for r in rooms:
        result.append(RoomBox(
            room_type=r.room_type,
            x_min=1.0 - r.x_max,
            y_min=r.y_min,
            x_max=1.0 - r.x_min,
            y_max=r.y_max,
        ))
    return result


def flip_vertical(rooms: List[RoomBox]) -> List[RoomBox]:
    """Flip layout vertically (mirror across X axis).

    y' = 1 - y, coordinates swap: y_min <-> y_max
    """
    result = []
    for r in rooms:
        result.append(RoomBox(
            room_type=r.room_type,
            x_min=r.x_min,
            y_min=1.0 - r.y_max,
            x_max=r.x_max,
            y_max=1.0 - r.y_min,
        ))
    return result


def rotate_90(rooms: List[RoomBox]) -> List[RoomBox]:
    """Rotate layout 90 degrees clockwise.

    (x, y) -> (y, 1-x)
    """
    result = []
    for r in rooms:
        # Original corners: (x_min, y_min), (x_max, y_max)
        # After 90° CW: x' = y, y' = 1-x
        new_x_min = r.y_min
        new_x_max = r.y_max
        new_y_min = 1.0 - r.x_max
        new_y_max = 1.0 - r.x_min
        result.append(RoomBox(
            room_type=r.room_type,
            x_min=new_x_min,
            y_min=new_y_min,
            x_max=new_x_max,
            y_max=new_y_max,
        ))
    return result


def rotate_180(rooms: List[RoomBox]) -> List[RoomBox]:
    """Rotate layout 180 degrees."""
    return flip_horizontal(flip_vertical(rooms))


def rotate_270(rooms: List[RoomBox]) -> List[RoomBox]:
    """Rotate layout 270 degrees clockwise (= 90 degrees counter-clockwise)."""
    return rotate_90(rotate_90(rotate_90(rooms)))


def jitter_coordinates(
    rooms: List[RoomBox],
    sigma: float = 0.01,
    clamp: float = 0.03,
    seed: Optional[int] = None
) -> List[RoomBox]:
    """Add small random perturbations to room coordinates.

    This helps the model learn invariance to small positional changes
    and prevents overfitting to exact coordinate values.

    Args:
        rooms: List of RoomBox to jitter
        sigma: Standard deviation of Gaussian noise
        clamp: Maximum jitter magnitude
        seed: Optional random seed for reproducibility

    Returns:
        Jittered rooms (maintains x_min < x_max, y_min < y_max)
    """
    if seed is not None:
        random.seed(seed)

    result = []
    for r in rooms:
        # Add jitter to each coordinate
        dx1 = max(-clamp, min(clamp, random.gauss(0, sigma)))
        dy1 = max(-clamp, min(clamp, random.gauss(0, sigma)))
        dx2 = max(-clamp, min(clamp, random.gauss(0, sigma)))
        dy2 = max(-clamp, min(clamp, random.gauss(0, sigma)))

        x_min = max(0.0, min(1.0, r.x_min + dx1))
        y_min = max(0.0, min(1.0, r.y_min + dy1))
        x_max = max(0.0, min(1.0, r.x_max + dx2))
        y_max = max(0.0, min(1.0, r.y_max + dy2))

        # Ensure valid box (min < max)
        if x_min >= x_max:
            x_min, x_max = r.x_min, r.x_max  # Revert if invalid
        if y_min >= y_max:
            y_min, y_max = r.y_min, r.y_max  # Revert if invalid

        result.append(RoomBox(
            room_type=r.room_type,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ))
    return result


def shuffle_room_order(rooms: List[RoomBox], seed: Optional[int] = None) -> List[RoomBox]:
    """Randomly shuffle the order of rooms.

    The autoregressive model learns dependencies on room order,
    but room order in layouts is arbitrary. Shuffling teaches
    the model that the same layout can have different orderings.

    Args:
        rooms: List of rooms to shuffle
        seed: Optional random seed

    Returns:
        Shuffled copy of rooms
    """
    if seed is not None:
        random.seed(seed)

    result = list(rooms)
    random.shuffle(result)
    return result


def augment_layout(
    rooms: List[RoomBox],
    config: Optional[AugmentationConfig] = None,
    seed: Optional[int] = None
) -> Tuple[List[RoomBox], List[str]]:
    """Apply random augmentations to a layout.

    Args:
        rooms: Original room list
        config: Augmentation config (uses defaults if None)
        seed: Optional random seed

    Returns:
        Tuple of (augmented_rooms, list_of_applied_augmentations)
    """
    if config is None:
        config = AugmentationConfig()

    if seed is not None:
        random.seed(seed)

    result = [RoomBox(r.room_type, r.x_min, r.y_min, r.x_max, r.y_max) for r in rooms]
    applied = []

    # Horizontal flip
    if config.enable_flip_h and random.random() < config.flip_prob:
        result = flip_horizontal(result)
        applied.append("flip_h")

    # Vertical flip
    if config.enable_flip_v and random.random() < config.flip_prob:
        result = flip_vertical(result)
        applied.append("flip_v")

    # Rotation (mutually exclusive: pick one rotation angle)
    if config.enable_rotate and random.random() < config.rotate_prob:
        rotation = random.choice([90, 180, 270])
        if rotation == 90:
            result = rotate_90(result)
        elif rotation == 180:
            result = rotate_180(result)
        else:
            result = rotate_270(result)
        applied.append(f"rotate_{rotation}")

    # Coordinate jitter
    if config.enable_jitter and random.random() < config.jitter_prob:
        result = jitter_coordinates(result, config.jitter_sigma, config.jitter_clamp)
        applied.append("jitter")

    # Room order shuffle
    if config.enable_shuffle and random.random() < config.shuffle_prob:
        result = shuffle_room_order(result)
        applied.append("shuffle")

    return result, applied


def expand_dataset_with_augmentations(
    samples: List[Tuple[List[RoomBox], str]],
    multiplier: int = 8,
    config: Optional[AugmentationConfig] = None
) -> List[Tuple[List[RoomBox], str]]:
    """Expand a dataset by generating augmented versions.

    Args:
        samples: List of (rooms, building_type) tuples
        multiplier: How many augmented versions per original
        config: Augmentation config

    Returns:
        Expanded list with original + augmented samples
    """
    if config is None:
        config = AugmentationConfig()

    expanded = []

    for rooms, building_type in samples:
        # Always include original
        expanded.append((rooms, building_type))

        # Generate augmented versions
        for i in range(multiplier - 1):
            aug_rooms, _ = augment_layout(rooms, config, seed=None)
            expanded.append((aug_rooms, building_type))

    return expanded


# Fixed augmentation variants for deterministic expansion
FIXED_TRANSFORMS = [
    ("original", lambda r: r),
    ("flip_h", flip_horizontal),
    ("flip_v", flip_vertical),
    ("flip_hv", lambda r: flip_horizontal(flip_vertical(r))),
    ("rotate_90", rotate_90),
    ("rotate_180", rotate_180),
    ("rotate_270", rotate_270),
    ("rotate_90_flip_h", lambda r: flip_horizontal(rotate_90(r))),
]


def deterministic_augment(rooms: List[RoomBox]) -> List[Tuple[List[RoomBox], str]]:
    """Generate all fixed transformation variants of a layout.

    Returns 8 variants: original + 3 rotations + flips + combinations.
    Useful for offline dataset expansion with reproducible results.
    """
    return [(transform(rooms), name) for name, transform in FIXED_TRANSFORMS]
