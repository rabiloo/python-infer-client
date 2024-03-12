from os import getenv
from os.path import dirname, join

import cv2
import numpy as np
import pytest

from infer_client.adapters.onnx import OnnxInferenceAdapter
from infer_client.adapters.triton import TritonInferenceAdapter

IMG_SIZE = 224
infer_onnx_obj = OnnxInferenceAdapter(
    model_name=join(dirname(__file__), "resources/test_classify"), version="1", limit_mem_gpu=-1
)
infer_triton_obj = TritonInferenceAdapter(
    triton_server=getenv("TRITON_SERVER"), model_name="test_classify", version="1", ssl=True
)


def preprocess(img, img_size):
    x = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
    x = (x / 255.0).astype(np.float32)
    return x.transpose(2, 0, 1)[None, ...]


def postprocess(output):
    e_x = np.exp(output)
    return np.argmax(e_x / e_x.sum())


@pytest.mark.parametrize(
    "path,expected",
    (
        (join(dirname(__file__), "resources/dog.jpg"), 264),
        (join(dirname(__file__), "resources/cat.jpg"), 285),
    ),
)
def test_inference(path: str, expected: str):
    img = cv2.imread(path)
    preprocessed_img = preprocess(img, (IMG_SIZE, IMG_SIZE))

    res_onnx = infer_onnx_obj.inference({"x": preprocessed_img}, ["400"])
    res_triton = infer_triton_obj.inference({"x": preprocessed_img}, ["400"])

    assert np.allclose(res_onnx, res_triton, rtol=1.0e-4)

    if not res_onnx:
        return None
    pred = res_onnx[0]
    output = postprocess(pred)
    assert output.item() == expected
