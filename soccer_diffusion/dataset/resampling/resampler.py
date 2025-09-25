from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from soccer_diffusion.dataset.imports.data import InputData

T = TypeVar("T")


@dataclass
class Sample(Generic[T]):
    data: T
    timestamp: float


class Resampler(ABC):
    @abstractmethod
    def resample(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        """
        Resample data from the input data DTO to a list of samples, where the relative timestamp of the input data
        is also the latest possible timestamp of any resampled data.
        Depending on the sampling strategy, the resulting resampled data and timestamps may or not be the same as the
        input data.

        Args:
            data (InputData): The input data DTO to resample from (actual data is one/many of the fields of the DTO)
            relative_timestamp (float): The relative timestamp of the input data (in seconds)

        Returns:
            list[Sample[InputData]]: A list of samples, where each sample contains resampled data and a timestamp
        """
        pass
