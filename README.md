# Python Infer Client

[![Testing](https://github.com/rabiloo/python-infer-client/actions/workflows/test.yml/badge.svg)](https://github.com/rabiloo/python-flysystem/actions/workflows/test.yml)
[![Latest Version](https://img.shields.io/pypi/v/flysystem.svg)](https://pypi.org/project/flysystem)
[![Downloads](https://img.shields.io/pypi/dm/flysystem.svg)](https://pypi.org/project/flysystem)
[![Pypi Status](https://img.shields.io/pypi/status/flysystem.svg)](https://pypi.org/project/flysystem)
[![Python Versions](https://img.shields.io/pypi/pyversions/flysystem.svg)](https://pypi.org/project/flysystem)

## About Python Infer Client

[Python Infer Client](https://github.com/rabiloo/ai_infer_client) is a python inference client library. It provides one interface to interact with many types of filesystems. When you use Flysystem, you're not only protected from vendor lock-in, you'll also have a consistent experience for which ever storage is right for you.

## Install

```
$ pip install infer-client
```

## Usage

```
from flysystem.adapters.local import LocalFilesystemAdapter
from flysystem.filesystem import Filesystem


adapter = LocalFilesystemAdapter(".")
filesystem = Filesystem(adapter)

filesystem.file_exists("/tmp/hello.txt")
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
