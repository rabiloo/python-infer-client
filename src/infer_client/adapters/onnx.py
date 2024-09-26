import sys

from os.path import join
from typing import Any, Dict, List

import onnxruntime

from ..adapters import InferenceAdapter


class OnnxInferenceAdapter(InferenceAdapter):
    def __init__(
        self,
        model_name: str,
        version: str = "1",
        limit_mem_gpu: int = -1,
        logger_level: int = 3,
        use_tf32: bool = True,
    ) -> None:
        onnxruntime.set_default_logger_severity(logger_level)
        providers = ["CPUExecutionProvider"]
        if onnxruntime.get_available_providers() == providers or limit_mem_gpu <= 0:
            # Run with CPU
            self.device_type = "cpu"
        elif limit_mem_gpu > 0:
            # Run with GPU
            self.device_type = "cuda"
            providers = [
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": limit_mem_gpu * 1024 * 1024 * 1024,
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": True,
                        "use_tf32": use_tf32,
                    },
                ),
                "CPUExecutionProvider",
            ]
        else:
            sys.exit("Not Support Device")
        if not model_name.endswith("model.onnx"):
            model_name = join(model_name, version, "model.onnx")
        self.ort_session = onnxruntime.InferenceSession(model_name, providers=providers)

    def health(self) -> bool:
        if self.ort_session is None:
            return False
        return True

    def inference(self, ort_inputs: Dict[str, Any], ort_out_names: List[str]) -> List[Any]:
        # IOBinding
        io_binding = self.ort_session.io_binding()
        for ort_input_name, ort_input in ort_inputs.items():
            io_binding.bind_cpu_input(ort_input_name, ort_input)
        for ort_out_name in ort_out_names:
            io_binding.bind_output(ort_out_name, self.device_type)

        self.ort_session.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        io_binding.clear_binding_inputs()
        io_binding.clear_binding_outputs()
        del io_binding

        return ort_outs
