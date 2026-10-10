"""ANE encoder adapter shape and mask contract without hardware or checkpoints."""
import numpy as np
import pytest
import torch

from laya.aneforge_encoder import AneEncoderModule


class StubEncoder:
    D = 4

    def __init__(self):
        self.inputs = []

    def hidden_states(self, ids, mask):
        self.inputs.append((ids.copy(), mask.copy()))
        return np.repeat(ids[:, :, None], self.D, axis=-1).astype(np.float32)


def test_ane_module_preserves_rows_and_masks():
    encoder = StubEncoder()
    module = AneEncoderModule(encoder, max_len=64, pad_token_id=9)
    ids = torch.tensor([[1, 2, 3], [4, 5, 9]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    output = module(ids, mask).last_hidden_state
    assert output.shape == (2, 3, 4)
    assert output[:, :, 0].tolist() == ids.tolist()
    assert len(encoder.inputs) == 2
    assert all(record[0].shape == (1, 48) for record in encoder.inputs)
    assert encoder.inputs[1][1][0, :4].tolist() == [1, 1, 0, 0]
    assert module.config.hidden_size == 4


def test_ane_module_rejects_overlong_input():
    module = AneEncoderModule(StubEncoder(), max_len=3)
    with pytest.raises(ValueError, match="max_len"):
        module(torch.zeros((1, 4), dtype=torch.long))
