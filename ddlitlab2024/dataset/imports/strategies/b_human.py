import concurrent.futures
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from pathlib import Path

from pybh.logs import Frame, Log, Record
from tqdm import tqdm

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.converters.game_state_converter import GameStateConverter
from ddlitlab2024.dataset.converters.image_converter import ImageConverter
from ddlitlab2024.dataset.converters.synced_data_converter import SyncedDataConverter
from ddlitlab2024.dataset.imports.data import ModelData
from ddlitlab2024.dataset.imports.model_importer import ImportMetadata, ImportStrategy
from ddlitlab2024.dataset.models import DEFAULT_IMG_SIZE, Recording

USED_REPRESENTATIONS: list[str] = [
    "FrameInfo",
    "GameControlData",
    "GameState",
    "InertialSensorData",
    "JointSensorData",
    "JPEGImage",
    "MotionRequest",
    "RobotPose",
    "StrategyStatus",
]


class SmartRecord(MutableMapping):
    def __init__(self, record) -> None:
        self.data = {key: record.__getattr__(key) for key in record}
        for key, value in self.data.items():
            if isinstance(value, Record):
                self.data[key] = SmartRecord(value)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class SmartFrame(MutableMapping):
    def __init__(self, frame: Frame):
        self.data = {repr: SmartRecord(frame[repr]) for repr in frame.representations if repr in USED_REPRESENTATIONS}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def time_ms(self) -> int | None:
        """Returns the time of the frame in milliseconds (probably since the start of the day), if available.

        :return: The time of the frame in milliseconds, if available.
        """
        if (info := self.get("FrameInfo")) is not None and (time := info.get("time")) is not None:
            return time

    @staticmethod
    def from_frame(frame: Frame) -> "SmartFrame":
        return SmartFrame(frame)


class BHumanImportStrategy(ImportStrategy):
    def __init__(
        self,
        metadata: ImportMetadata,
        image_converter: ImageConverter,
        game_state_converter: GameStateConverter,
        synced_data_converter: SyncedDataConverter,
    ):
        self.metadata = metadata
        self.image_converter = image_converter
        self.game_state_converter = game_state_converter
        self.synced_data_converter = synced_data_converter

        self.model_data = ModelData()

    def convert_to_model_data(self, file_path: Path) -> ModelData:
        self.model_data.recording = self._create_recording(file_path)

        log = self._read_log_file(file_path)
        orig_frames: list[tuple[int, Frame]] = [
            (i, frame) for i, frame in tqdm(enumerate(log), desc="Reading frames", unit="frames", total=len(log))
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results: list[tuple[int, SmartFrame]] = list(
                tqdm(
                    executor.map(lambda x: (x[0], SmartFrame.from_frame(x[1])), orig_frames),
                    desc="Converting frames",
                    unit="frames",
                    total=len(orig_frames),
                )
            )
        del orig_frames
        frames: list[SmartFrame] = [frame for _, frame in sorted(results, key=lambda x: x[0])]
        del results

        self._populate_timestamps(frames)

        return self.model_data

    def _read_log_file(self, file_path: Path) -> Log:
        logger.debug(f"Reading B-Human data from file '{file_path}'...")
        log = Log(str(file_path), keep_going=True)
        logger.debug(
            f"Read log with {len(log)} Frames: {log.bodyName=}, {log.headName=}, {log.identifier=}, {log.location=}, {log.playerNumber=}, {log.scenario=}, {log.suffix=}"
        )
        return log

    def _create_recording(self, file_path: Path) -> Recording:
        recording = Recording(
            allow_public=self.metadata.allow_public,
            original_file=file_path.name,
            team_name=self.metadata.team_name,
            team_color=None,  # Needs to be overwritten when processing GameStates
            robot_type=self.metadata.robot_type,
            start_time=None,
            end_time=None,
            location=self.metadata.location,
            simulated=self.metadata.simulated,
            img_width=DEFAULT_IMG_SIZE[0],
            img_height=DEFAULT_IMG_SIZE[1],
            img_width_scaling=0.0,  # needs to be overwritten when processing images
            img_height_scaling=0.0,  # needs to be overwritten when processing images
        )
        return recording

    def _extract_timeframe(self, frames: list[SmartFrame]) -> tuple[datetime, datetime]:
        first_time_ms: int
        for frame in frames:
            if (time_ms := frame.time_ms()) is not None:
                first_time_ms = time_ms
                break

        logger.debug(f"Timestamp offset: {first_time_ms} [ms]")

        last_time_ms: int
        for frame in reversed(frames):
            if (time_ms := frame.time_ms()) is not None:
                last_time_ms = time_ms
                break

        assert isinstance(first_time_ms, int) and isinstance(last_time_ms, int), "Timestamps (ms) must be integers"
        assert first_time_ms <= last_time_ms, "First timestamp must be before last timestamp"

        first_timestamp: datetime = datetime(2024, 5, 3)  # TODO: Replace with actual timestamp
        last_timestamp: datetime = datetime(2024, 5, 3) + timedelta(
            milliseconds=last_time_ms - first_time_ms
        )  # TODO: Replace with actual timestamp

        assert first_timestamp <= last_timestamp, "First timestamp must be before last timestamp"

        return first_timestamp, last_timestamp

    def _populate_timestamps(self, frames: list[SmartFrame]) -> None:
        first_timestamp, last_timestamp = self._extract_timeframe(frames)

        assert self.model_data.recording is not None, "Recording must be defined to populate timestamps"
        self.model_data.recording.start_time = first_timestamp
        self.model_data.recording.end_time = last_timestamp
