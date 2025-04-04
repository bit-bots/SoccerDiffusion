from abc import ABC, abstractmethod
from pathlib import Path

from ddlitlab2024.dataset.converters.converter import Converter
from ddlitlab2024.dataset.db import Database
from ddlitlab2024.dataset.imports.data import ImportMetadata, ModelData


class ImportStrategy(ABC):
    def __init__(
        self,
        metadata: ImportMetadata,
        image_converter: Converter,
        game_state_converter: Converter,
        synced_data_converter: Converter,
    ):
        self.metadata = metadata
        self.image_converter = image_converter
        self.game_state_converter = game_state_converter
        self.synced_data_converter = synced_data_converter

    @abstractmethod
    def convert_to_model_data(self, file_path: Path) -> ModelData:
        pass


class ModelImporter:
    def __init__(self, db: Database, strategy: ImportStrategy):
        self.db = db
        self.strategy = strategy

    def import_to_db(self, file_path: Path):
        model_data: ModelData = self.strategy.convert_to_model_data(file_path)

        required_fields = ["images", "game_states", "joint_states", "joint_commands", "rotations"]
        for field in required_fields:
            if not len(getattr(model_data, field)):
                raise ValueError(f"No {field} models extracted from the file, aborting import.")

        self.db.session.add_all(model_data.model_instances())
        self.db.session.commit()
