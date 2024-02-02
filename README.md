# Python Infer Client

[![Testing](https://github.com/rabiloo/python-infer-client/actions/workflows/test.yml/badge.svg)](https://github.com/rabiloo/python-infer-client/actions/workflows/test.yml)
[![Latest Version](https://img.shields.io/pypi/v/infer-client.svg)](https://pypi.org/project/infer-client)
[![Downloads](https://img.shields.io/pypi/dm/infer-client.svg)](https://pypi.org/project/infer-client)
[![Pypi Status](https://img.shields.io/pypi/status/infer-client.svg)](https://pypi.org/project/infer-client)
[![Python Versions](https://img.shields.io/pypi/pyversions/infer-client.svg)](https://pypi.org/project/infer-client)

## About Python Infer Client

[Python Infer Client](https://github.com/rabiloo/python-infer-client) is a python inference client library. It provides one interface to interact with many types of inference client as onnxruntime, tritonclient...

## Install
With using the tritonclient client, only supported with GRPC
```
$ pip install infer-client[tritonclient]
```

With using the onnxruntime client, both CPU and GPU are supported
```
$ pip install infer-client[onnxruntime]
or
$ pip install infer-client[onnxruntime-gpu]
```
## Usage

```
import numpy as np

from infer_client.adapters.onnx import OnnxInferenceAdapter
from infer_client.inference import Inference


adapter = OnnxInferenceAdapter(model_name="resources/test_classify", version="1", limit_mem_gpu=-1)
infer_client_obj = Inference(adapter)

res = infer_client_obj.inference({"input": np.random.rand(1, 3, 224, 224)}, ["output"])
```

## Changelog

Please see [CHANGELOG](CHANGELOG.md) for more information on what has changed recently.

## Contributing

Please see [CONTRIBUTING](.github/CONTRIBUTING.md) for details.

## Security Vulnerabilities

Please review [our security policy](../../security/policy) on how to report security vulnerabilities.

## Credits

- [Dao Quang Duy](https://github.com/duydq12)
- [All Contributors](../../contributors)

## License

The MIT License (MIT). Please see [License File](LICENSE) for more information.

## Reference
- [Onnxruntime](https://github.com/microsoft/onnxruntime)
- [Triton Client](https://github.com/triton-inference-server/client)
