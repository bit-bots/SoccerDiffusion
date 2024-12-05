from abc import ABC, abstractmethod

from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.models import Recording
from ddlitlab2024.dataset.resampling.resampler import Resampler


class Converter(ABC):
    def __init__(self, resampler: Resampler) -> None:
        self.resampler = resampler

    @abstractmethod
    def populate_recording_metadata(self, data: InputData, recording: Recording):
        """
        Different converters of specific data/topics might need to extract
        information about a recording in general and update its metadata
        e.g. from a bitbots /gamestate message we extract the team's color

        Args:
            data: The input data to extract metadata from (e.g. a gamestate message)
            recording: The recording db model to update
        """
        pass

    @abstractmethod
    def convert_to_model(self, data: InputData, relative_timestamp: float, recording: Recording) -> ModelData:
        """
        Convert the input data to our domain model that can be stored in the database and later used for training.
        The model is contained in a ModelData DTO, for generic handling of different models.

        Args:
            data (InputData): The input data to convert to a model (e.g. a gamestate ros message)
            relative_timestamp (float): The timestamp of the data relative to the start of the recording
            recording (Recording): The recording db model the created model will be associated with

        Returns:
            ModelData: Dataclass containing list of models to be created from the data (fields can be empty)
        """
        pass
