"""ANE encoder adapter shape and mask contract without hardware or checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest
import torch

from laya.aneforge_encoder import AneEncoderModule


class StubEncoder:
    D = 4

    def __init__(self):
        self.inputs = []
        self.releases = 0

    def hidden_states(self, ids, mask):
        self.inputs.append((ids.copy(), mask.copy()))
        return np.repeat(ids[:, :, None], self.D, axis=-1).astype(np.float32)

    def release(self):
        self.releases += 1


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


def test_ane_module_releases_old_bucket_before_compiling_new_one():
    encoder = StubEncoder()
    module = AneEncoderModule(encoder, max_len=128)
    for length in (96, 118, 118, 96):
        output = module(torch.ones((1, length), dtype=torch.long)).last_hidden_state
        assert output.shape == (1, length, encoder.D)
    assert [ids.shape[1] for ids, _ in encoder.inputs] == [96, 128, 128, 96]
    assert encoder.releases == 2


def test_ane_module_keeps_parallel_requests_separate():
    class SharedBufferEncoder:
        D = 1

        def __init__(self):
            self.output = np.empty((1, 48, 1), dtype=np.float32)
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()

        def hidden_states(self, ids, mask):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                self.output.fill(float(ids[0, 0]))
                time.sleep(0.05)
                return self.output
            finally:
                with self.lock:
                    self.active -= 1

    encoder = SharedBufferEncoder()
    module = AneEncoderModule(encoder, max_len=64)
    start = threading.Barrier(3)

    def predict(value):
        start.wait()
        return module(torch.tensor([[value, 0, 0]])).last_hidden_state[0, 0, 0].item()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(predict, 1)
        second = pool.submit(predict, 2)
        start.wait()
        assert (first.result(), second.result()) == (1.0, 2.0)
    assert encoder.peak == 1
