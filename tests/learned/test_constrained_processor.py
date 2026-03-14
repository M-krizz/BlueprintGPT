import pytest
import torch
from learned.model.sample import SpecConstrainedProcessor
from learned.data.tokenizer_layout import LayoutTokenizer

def test_spec_constrained_processor():
    tok = LayoutTokenizer(num_bins=256)
    spec = {"rooms": [{"type": "Bedroom"}, {"type": "Bedroom"}, {"type": "Kitchen"}]}
    proc = SpecConstrainedProcessor(tok, spec, max_rooms=5)
    
    # Simulate: after ROOM_TOKEN, logits for non-spec types should be -inf
    fake_logits = torch.zeros(1, tok.vocab_size)
    fake_seq = torch.tensor([[1, 31, 3]])  # BOS, cond, ROOM_TOKEN
    
    result = proc(fake_logits, fake_seq)
    
    bedroom_tok = tok._type2tok["Bedroom"]
    kitchen_tok = tok._type2tok["Kitchen"]
    living_tok = tok._type2tok["LivingRoom"]
    
    assert result[0, living_tok].item() == float("-inf"), "LivingRoom should be masked out since it's not in spec"
    assert result[0, bedroom_tok].item() == 0.0, "Bedroom should be allowed"
    assert result[0, kitchen_tok].item() == 0.0, "Kitchen should be allowed"
