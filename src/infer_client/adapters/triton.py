import sys
import uuid

from typing import Any, Dict, List

from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype

from ..adapters import InferenceAdapter
from ..error import UnableToInference
from .tritonclient_grpc_custom import InferenceServerClientCustom


class TritonInferenceAdapter(InferenceAdapter):
    def __init__(
        self,
        triton_server: str,
        model_name: str,
        version: str = "1",
        client_timeout: int = 30,
        ssl: bool = False,
        options: Dict[str, Any] = None,
    ) -> None:
        if options is None:
            options = [
                ("grpc.max_send_message_length", 512 * 1024 * 1024),
                ("grpc.max_receive_message_length", 512 * 1024 * 1024),
            ]

        self.grpc_client = InferenceServerClientCustom(url=triton_server, ssl=ssl, channel_args=options)
        if not (
            self.grpc_client.is_server_live(client_timeout=client_timeout)
            and self.grpc_client.is_server_ready(client_timeout=client_timeout)
        ):
            sys.exit("Triton server is not live")

        self.model_name = model_name
        self.version = version
        self.client_timeout = client_timeout

    def health(self) -> bool:
        if self.grpc_client.is_model_ready(self.model_name, self.version, client_timeout=self.client_timeout):
            return True
        return False

    def inference(self, ort_inputs: Dict[str, Any], ort_out_names: List[str]) -> List[Any]:
        inputs = []
        for input_name, input_data in ort_inputs.items():
            infer_input = InferInput(
                name=input_name, shape=input_data.shape, datatype=np_to_triton_dtype(input_data.dtype)
            )
            infer_input.set_data_from_numpy(input_data)
            inputs.append(infer_input)

        outputs = [InferRequestedOutput(output_name) for output_name in ort_out_names]

        try:
            res = self.grpc_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                model_version=self.version,
                outputs=outputs,
                request_id=uuid.uuid4().__str__(),
                client_timeout=self.client_timeout,
            )
        except Exception as ex:
            raise UnableToInference.with_model(self.model_name, self.version, str(ex))

        if len(ort_out_names) == 1:
            return [res.as_numpy(ort_out_names[0])]
        return [res.as_numpy(out) for out in ort_out_names]
