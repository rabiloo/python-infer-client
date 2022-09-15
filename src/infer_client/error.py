"""
Module flysystem
"""

from typing import final

from typing_extensions import Self


class InferenceException(Exception):
    """
    Base exception class for AI inference client package
    """


@final
class UnableToInference(InferenceException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self._model_name = ""
        self._model_version = ""
        self._reason = ""

    @final
    def model_name(self) -> str:
        return self._model_name

    @final
    def model_version(self) -> str:
        return self._model_version

    @final
    def reason(self) -> str:
        return self._reason

    @classmethod
    def with_model(cls, model_name: str, model_version: str, reason: str = "") -> Self:
        msg = f"Unable to inference with model: {model_name}-{model_version}. {reason}".rstrip()
        this = cls(msg)
        this._model_name = model_name
        this._model_version = model_version
        this._reason = reason
        return this
