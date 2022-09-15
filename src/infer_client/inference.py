from abc import ABCMeta
from typing import Any, Dict

from .adapters import InferenceAdapter


class Inference(metaclass=ABCMeta):
    """
    Inference Client
    """

    def __init__(
        self,
        adapter: InferenceAdapter,
        config: Dict[str, Any] = None,
    ):
        self.adapter = adapter
        self.config = config

    def health(self) -> bool:
        return self.adapter.health()

    def inference(self, ort_inputs, ort_out_names):
        return self.adapter.inference(ort_inputs, ort_out_names)
