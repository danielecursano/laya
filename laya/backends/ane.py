"""Backend adapter for Laya's ANEForge encoder and CPU decision head."""
from .base import Backend, BackendUnavailable


class AneBackend(Backend):
    name = "ane"

    def _install(self):
        from ..aneforge_encoder import is_ane_available, AneEncoderModule
        if not is_ane_available():
            raise BackendUnavailable("aneforge is not installed or ANE is unavailable")
        if not isinstance(self.model.encoder, AneEncoderModule):
            raise BackendUnavailable("load the agent with backend='ane' to build its ANE encoder")
        self.max_len = self.model.encoder.max_len
