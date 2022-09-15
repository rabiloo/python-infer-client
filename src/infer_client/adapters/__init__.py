from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List


class InferenceAdapter(metaclass=ABCMeta):
    """
    FilesystemAdapter interface
    """

    @abstractmethod
    def health(self) -> bool:
        """
        Determine if model is running.
        Arguments:
        Returns:
            True if the file existed
        """

    @abstractmethod
    def inference(self, ort_inputs: Dict[str, Any], ort_out_names: List[str]) -> List[Any]:
        """
        Determine if a directory exists.
        Arguments:
            ort_inputs: The dictionary of input
            ort_out_names: List of name of nodes wanna get value
        Returns:
            List of value of nodes
        """
