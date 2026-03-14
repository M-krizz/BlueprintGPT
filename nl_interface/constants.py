"""Constants and helpers for the additive natural-language spec layer."""

from __future__ import annotations

import json
from pathlib import Path

ALLOWED_BUILDING_TYPE = "Residential"

ALLOWED_PLOT_TYPES = ("5Marla", "10Marla", "20Marla", "Custom")
ALLOWED_ENTRANCE_SIDES = ("North", "South", "East", "West")
ALLOWED_ROOM_TYPES = (
    "Bedroom",
    "Bathroom",
    "Kitchen",
    "DrawingRoom",
    "DiningRoom",
    "Garage",
    "Store",
)
ALLOWED_RELATIONSHIPS = ("near_to", "adjacent_to", "far_from")
ALLOWED_PRIVACY_ZONES = ("public", "service", "private")

DEFAULT_WEIGHTS = {
    "privacy": 1.0 / 3.0,
    "compactness": 1.0 / 3.0,
    "corridor": 1.0 / 3.0,
}

DEFAULT_PRIVACY_BY_ROOM = {
    "Bedroom": "private",
    "Bathroom": "private",
    "Kitchen": "service",
    "DrawingRoom": "public",
    "DiningRoom": "public",
    "Garage": "service",
    "Store": "service",
}

EXTERNAL_TO_INTERNAL_ROOM = {
    "Bedroom": "Bedroom",
    "Bathroom": "Bathroom",
    "Kitchen": "Kitchen",
    "DrawingRoom": "LivingRoom",
    "DiningRoom": "DiningRoom",
    "Garage": "Garage",
    "Store": "Store",
}

CORE_ALGORITHMIC_ROOM_TYPES = {
    "Bedroom",
    "Bathroom",
    "Kitchen",
    "DrawingRoom",
}

EXTENDED_LEARNED_ROOM_TYPES = {
    "DiningRoom",
    "Garage",
    "Store",
}

NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

ROOM_LABELS = {
    "Bedroom": ("bedroom", "bedrooms", "bed", "beds"),
    "Bathroom": ("bathroom", "bathrooms", "bath", "baths"),
    "Kitchen": ("kitchen", "kitchens"),
    "DrawingRoom": ("drawing room", "drawing rooms", "drawingroom", "drawingrooms"),
    "DiningRoom": ("dining room", "dining rooms", "diningroom", "diningrooms"),
    "Garage": ("garage", "garages"),
    "Store": ("store", "stores"),
}

UNSUPPORTED_ROOM_LABELS = {
    "living room": "LivingRoom",
    "living rooms": "LivingRoom",
    "livingroom": "LivingRoom",
    "toilet": "WC",
    "toilets": "WC",
    "wc": "WC",
    "office": "Office",
    "study": "Study",
    "lobby": "Lobby",
    "stairs": "Stairs",
    "passage": "Passage",
}

UNSUPPORTED_RELATION_PHRASES = (
    "connected to",
    "opposite",
    "across from",
)

RELATION_PHRASES = {
    "adjacent_to": (
        "adjacent to",
        "next to",
        "beside",
        "shares a wall with",
    ),
    "near_to": (
        "near to",
        "near",
        "close to",
    ),
    "far_from": (
        "far from",
        "away from",
        "separate from",
    ),
}

STYLE_HINTS = (
    {
        "phrases": ("open plan", "open layout", "feel open", "more open"),
        "delta": {"privacy": -0.18, "compactness": 0.09, "corridor": 0.09},
        "note": "I lowered the privacy weight to allow for a more open plan.",
    },
    {
        "phrases": ("privacy first", "more privacy", "private feel", "private layout"),
        "delta": {"privacy": 0.24, "compactness": -0.12, "corridor": -0.12},
        "note": "I increased the privacy weight to protect quieter rooms from public areas.",
    },
    {
        "phrases": ("compact", "space efficient", "efficient layout", "compact layout"),
        "delta": {"privacy": -0.06, "compactness": 0.18, "corridor": -0.12},
        "note": "I increased the compactness weight to favor a tighter layout.",
    },
    {
        "phrases": ("minimize corridor", "minimal corridor", "reduce corridor", "less corridor", "fewer corridors"),
        "delta": {"privacy": -0.05, "compactness": 0.05, "corridor": 0.20},
        "note": "I increased the corridor weight to reduce circulation waste.",
    },
)

PLOT_CAPACITY_PATH = Path(__file__).with_name("plot_capacity.json")
DEFAULT_PLOT_CAPACITY = {
    "5Marla": {"max_total_rooms": 5, "max_bedrooms": 3},
    "10Marla": {"max_total_rooms": 8, "max_bedrooms": 5},
    "20Marla": {"max_total_rooms": 12, "max_bedrooms": 7},
    "Custom": {},
}


def load_plot_capacity_config():
    if not PLOT_CAPACITY_PATH.exists():
        return dict(DEFAULT_PLOT_CAPACITY)
    try:
        with open(PLOT_CAPACITY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return dict(DEFAULT_PLOT_CAPACITY)
