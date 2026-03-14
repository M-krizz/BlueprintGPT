import pytest
from learned.data.tokenizer_layout import LayoutTokenizer, RoomBox

def test_tokenizer_encode_decode():
    tok = LayoutTokenizer(num_bins=256)
    
    r1 = RoomBox("Bedroom", 0.1, 0.2, 0.5, 0.6)
    r2 = RoomBox("Kitchen", 0.5, 0.2, 0.9, 0.6)
    
    seq = tok.encode_sample([r1, r2], building_type="Residential")
    assert len(seq) > 0
    
    rooms = tok.decode_rooms(seq)
    assert len(rooms) == 2
    assert rooms[0].room_type == "Bedroom"
    assert rooms[1].room_type == "Kitchen"
    
def test_tokenizer_pad():
    tok = LayoutTokenizer(num_bins=256)
    padded = tok.pad([1, 2, 3], max_len=5)
    assert padded == [1, 2, 3, 0, 0]
