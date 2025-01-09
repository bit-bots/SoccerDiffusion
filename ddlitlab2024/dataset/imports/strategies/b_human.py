from pathlib import Path

from ddlitlab2024.dataset.converters.game_state_converter import GameStateConverter
from ddlitlab2024.dataset.converters.image_converter import ImageConverter
from ddlitlab2024.dataset.converters.synced_data_converter import SyncedDataConverter
from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.imports.model_importer import ImportMetadata, ImportStrategy

USED_REPRESENTATIONS: list[str] = []


class BHumanImportStrategy(ImportStrategy):
    def __init__(
        self,
        metadata: ImportMetadata,
        image_converter: ImageConverter,
        game_state_converter: GameStateConverter,
        synced_data_converter: SyncedDataConverter,
    ):
        self.image_converter = image_converter
        self.game_state_converter = game_state_converter
        self.synced_data_converter = synced_data_converter

        self.model_data = ModelData()

    def convert_to_model_data(self, file_path: Path) -> ModelData:
        pass
