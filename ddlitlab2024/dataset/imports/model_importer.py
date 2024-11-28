from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ddlitlab2024.dataset.db import Database
from ddlitlab2024.dataset.models import GameState, Image, JointCommands, JointStates, Recording


@dataclass
class ImportMetadata:
    allow_public: bool
    team_name: str
    robot_type: str
    location: str
    simulated: bool


@dataclass
class Sample[T]:
    data: T
    timestamp: float
    was_sampled_already: bool = False


@dataclass
class InputData:
    image: Any = None
    game_state: Any = None
    joint_state: Any = None
    joint_command: Any = None


@dataclass
class ModelData:
    recording: Recording | None = None
    game_states: list[GameState] = field(default_factory=list)
    joint_states: list[JointStates] = field(default_factory=list)
    joint_commands: list[JointCommands] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)

    def model_instances(self):
        return [self.recording] + self.game_states + self.joint_states + self.joint_commands + self.images

    def merge(self, other: "ModelData") -> "ModelData":
        self.game_states.extend(other.game_states)
        self.joint_states.extend(other.joint_states)
        self.joint_commands.extend(other.joint_commands)
        self.images.extend(other.images)
        return self


class ImportStrategy(ABC):
    @abstractmethod
    def convert_to_model_data(self, file_path: Path) -> ModelData:
        pass


class ModelImporter:
    def __init__(self, db: Database, strategy: ImportStrategy):
        self.db = db
        self.strategy = strategy

    def import_to_db(self, file_path: Path):
        model_data: ModelData = self.strategy.convert_to_model_data(file_path)
        self.db.session.add_all(model_data.model_instances())
        self.db.session.commit()
