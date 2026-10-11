"""ANEForge ModernBERT encoder with Laya's decision head on CPU."""
import threading
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


# ANEForge programs reuse input and output buffers, including across requests from
# FastAPI's thread pool. Serialize dispatch for all ANE-backed Laya agents.
_ANE_DISPATCH_LOCK = threading.Lock()


def is_ane_available():
    try:
        from aneforge._runtime import _find_dylib
        _find_dylib()
        return True
    except (ImportError, OSError, RuntimeError):
        return False


class AneEncoderModule(nn.Module):
    """Adapt ANEForge token states to the HF encoder output used by DecisionModel."""

    def __init__(self, encoder, max_len, pad_token_id=0):
        super().__init__()
        self.ane_encoder = encoder
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self._active_bucket = None
        self.config = SimpleNamespace(hidden_size=encoder.D, model_type="modernbert", reference_compile=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        rows, length = input_ids.shape
        if length > self.max_len:
            raise ValueError("ANE encoder max_len=%d; request has %d tokens" % (self.max_len, length))
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        ids = input_ids.detach().cpu().numpy()
        masks = attention_mask.detach().cpu().numpy()
        outputs = []
        # ANEForge compiles by sequence length; sharing a 16-token bucket avoids a graph
        # for every different request while keeping compilation smaller than max_len.
        bucket = min(self.max_len, max(48, ((length + 15) // 16) * 16))
        with _ANE_DISPATCH_LOCK:
            # ModernBERT-large holds many ANE programs per bucket. Keeping the old bucket
            # resident can exhaust ANE resources while compiling a new one (op_create err=13).
            if self._active_bucket is not None and bucket != self._active_bucket:
                self.ane_encoder.release()
                self._active_bucket = None
            for row in range(rows):
                padded_ids = np.full((1, bucket), self.pad_token_id, dtype=np.int64)
                padded_mask = np.zeros((1, bucket), dtype=np.int64)
                padded_ids[0, :length] = ids[row]
                padded_mask[0, :length] = masks[row]
                states = self.ane_encoder.hidden_states(padded_ids, padded_mask)
                outputs.append(np.array(states[0, :length], dtype=np.float32, copy=True))
            if rows:
                self._active_bucket = bucket
        result = np.stack(outputs) if outputs else np.empty((0, length, self.ane_encoder.D), np.float32)
        return SimpleNamespace(last_hidden_state=torch.from_numpy(result.copy()).to(input_ids.device))


def build_ane_model(cfg, weights, encoder_dir=None, pad_token_id=0, tokenizer=None):
    """Build a Laya model from its checkpoint with ANE token states and CPU heads."""
    from aneforge.models import Encoder
    from transformers import AutoConfig
    from .common import DecisionModel

    name = encoder_dir or cfg["encoder"]
    enc_cfg = AutoConfig.from_pretrained(name)
    if enc_cfg.model_type != "modernbert":
        raise ValueError("ANE backend supports ModernBERT checkpoints only")
    encoder = Encoder(name, state_dict=weights, config=enc_cfg, tokenizer=tokenizer, chunk_layers=1)
    max_len = int(cfg.get("max_len", 512))
    module = AneEncoderModule(encoder, max_len, pad_token_id)
    model = DecisionModel(module, head_layers=cfg.get("head_layers", 2),
                          n_act=len(cfg.get("act_costs", {})) + 1, no_init=True)
    head_weights = {key: value for key, value in weights.items() if not key.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(head_weights, strict=False)
    if missing or unexpected:
        raise ValueError("incomplete Laya decision head: missing=%s unexpected=%s" % (missing, unexpected))
    return model
