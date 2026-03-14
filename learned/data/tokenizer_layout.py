"""
tokenizer_layout.py – Discretize normalised coordinates into integer tokens.

Vocabulary layout
-----------------
Token 0                : <PAD>
Token 1                : <BOS>  (beginning of sequence)
Token 2                : <EOS>  (end of sequence)
Token 3                : <ROOM> (room separator)
Tokens 4 .. 4+T-1     : room-type tokens
Tokens 4+T .. 4+T+C-1 : condition tokens (building type / plot size)
Tokens BASE .. BASE+B-1: coordinate bins (0 … B-1 mapped to [0, 1])

Default B = 256  →  ~1 mm resolution on a 25 m plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Lightweight RoomBox (no external dep so visualiser can import standalone) ──

@dataclass
class RoomBox:
    """Axis-aligned bounding box for a single room."""
    room_type: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
ROOM_TOKEN = 3
SPECIAL_COUNT = 4

# ── Room types (original pipeline + Kaggle mazharrehan/floorplan) ─────────────
ROOM_TYPES: List[str] = [
    "Bedroom",
    "LivingRoom",
    "Kitchen",
    "Bathroom",
    "WC",
    "DiningRoom",
    "Study",
    "Storage",
    "Balcony",
    "DrawingRoom",
    "Garage",
    "Lounge",
    "Lobby",
    "Passage",
    "Stairs",
    "Lawn",
    "OpenSpace",
    "Staircase",
    "SideGarden",
    "Dining",
    "DressingArea",
    "Store",
    "PrayerRoom",
    "ServantQuarter",
    "Backyard",
    "Laundry",
    "Unknown",
]

# ── Building / condition types ────────────────────────────────────────────────
BUILDING_TYPES: List[str] = [
    "Residential",
    "Commercial",
    "Mixed",
    "5Marla",
    "10Marla",
    "20Marla",
]

DEFAULT_NUM_BINS = 256


# ===========================================================================
#  LayoutTokenizer
# ===========================================================================

@dataclass
class LayoutTokenizer:
    """Convert between RoomBox lists and integer token sequences."""

    num_bins: int = DEFAULT_NUM_BINS

    _type_offset: int = field(init=False, repr=False)
    _cond_offset: int = field(init=False, repr=False)
    _coord_offset: int = field(init=False, repr=False)
    _vocab_size: int = field(init=False, repr=False)
    _type2tok: Dict[str, int] = field(init=False, repr=False)
    _tok2type: Dict[int, str] = field(init=False, repr=False)
    _btype2tok: Dict[str, int] = field(init=False, repr=False)
    _tok2btype: Dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        self._type_offset = SPECIAL_COUNT
        self._cond_offset = self._type_offset + len(ROOM_TYPES)
        self._coord_offset = self._cond_offset + len(BUILDING_TYPES)
        self._vocab_size = self._coord_offset + self.num_bins

        self._type2tok = {t: self._type_offset + i for i, t in enumerate(ROOM_TYPES)}
        self._tok2type = {v: k for k, v in self._type2tok.items()}
        self._btype2tok = {t: self._cond_offset + i for i, t in enumerate(BUILDING_TYPES)}
        self._tok2btype = {v: k for k, v in self._btype2tok.items()}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def coord_offset(self) -> int:
        return self._coord_offset

    @property
    def coord_token_end(self) -> int:
        return self._coord_offset + self.num_bins

    @property
    def type_token_start(self) -> int:
        return self._type_offset

    @property
    def type_token_end(self) -> int:
        return self._cond_offset

    def _is_coord_token(self, token: int) -> bool:
        return self._coord_offset <= token < self.coord_token_end

    def _bin(self, value: float) -> int:
        b = int(round(max(0.0, min(1.0, value)) * (self.num_bins - 1)))
        return self._coord_offset + b

    def _unbin(self, token: int) -> float:
        return (token - self._coord_offset) / max(self.num_bins - 1, 1)

    def encode_room(self, room: RoomBox) -> List[int]:
        type_tok = self._type2tok.get(room.room_type, self._type2tok["Unknown"])
        return [ROOM_TOKEN, type_tok,
                self._bin(room.x_min), self._bin(room.y_min),
                self._bin(room.x_max), self._bin(room.y_max)]

    def encode_condition(self, building_type: str = "Residential",
                         room_types: Optional[List[str]] = None) -> List[int]:
        tokens: list[int] = [BOS_TOKEN]
        btok = self._btype2tok.get(building_type, self._btype2tok["Residential"])
        tokens.append(btok)
        if room_types:
            for rt in room_types:
                tokens.append(self._type2tok.get(rt, self._type2tok["Unknown"]))
        return tokens

    def encode_sample(self, rooms: List[RoomBox],
                      building_type: str = "Residential",
                      room_order: Optional[List[int]] = None) -> List[int]:
        cond = self.encode_condition(building_type, [r.room_type for r in rooms])
        seq = list(cond)
        order = room_order if room_order is not None else list(range(len(rooms)))
        for idx in order:
            seq.extend(self.encode_room(rooms[idx]))
        seq.append(EOS_TOKEN)
        return seq

    def decode_rooms(self, tokens: List[int]) -> List[RoomBox]:
        rooms: list[RoomBox] = []
        i = 0
        while i < len(tokens):
            if tokens[i] == ROOM_TOKEN:
                if i + 5 >= len(tokens):
                    break
                type_tok = tokens[i + 1]
                coord_toks = tokens[i + 2:i + 6]
                if not all(self._is_coord_token(t) for t in coord_toks):
                    # Skip malformed room chunks (e.g., model emitted <ROOM>
                    # where a coordinate token should be).
                    i += 1
                    continue
                x1 = self._unbin(coord_toks[0])
                y1 = self._unbin(coord_toks[1])
                x2 = self._unbin(coord_toks[2])
                y2 = self._unbin(coord_toks[3])
                rtype = self._tok2type.get(type_tok, "Unknown")
                rooms.append(RoomBox(rtype, min(x1, x2), min(y1, y2),
                                     max(x1, x2), max(y1, y2)))
                i += 6
            elif tokens[i] == EOS_TOKEN:
                break
            else:
                i += 1
        return rooms

    def decode_building_type(self, tokens: List[int]) -> str:
        for t in tokens:
            if t in self._tok2btype:
                return self._tok2btype[t]
        return "Residential"

    def pad(self, seq: List[int], max_len: int) -> List[int]:
        return seq[:max_len] + [PAD_TOKEN] * max(0, max_len - len(seq))
