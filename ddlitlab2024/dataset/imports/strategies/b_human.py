import io
import sys
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from dateutil.parser import ParserError
from dateutil.parser import parse as dateutil_parse
from PIL import Image
from pybh.logs import Array, Frame, Log, Record
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
        self.data = {}
        for key in record:
            value = record.__getattr__(key)
            match value:
                case Record():
                    self.data[key] = SmartRecord(value)
                case Array():
                    array_values = []
                    for array_value in value:
                        match array_value:
                            case Record():
                                array_values.append(SmartRecord(array_value))
                            case _:
                                array_values.append(array_value)
                    self.data[key] = array_values
                case _:
                    self.data[key] = value

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

    def get(self, key, default=None):
        return self.data.get(key, default)


class SmartFrame(MutableMapping):
    def __init__(self, frame: Frame):
        self.data = {repr: SmartRecord(frame[repr]) for repr in frame.representations if repr in USED_REPRESENTATIONS}
        self.thread: str = frame.thread

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

    def get(self, key, default=None):
        return self.data.get(key, default)

    @staticmethod
    def from_frame(frame: Frame) -> "SmartFrame":
        return SmartFrame(frame)

    def time_ms(self) -> int | None:
        """Returns the time of the frame in milliseconds since the start of the log, if available.

        :return: The time of the frame in milliseconds, if available.
        """
        if (info := self.get("FrameInfo")) is not None and (time := info.get("time")) is not None:
            # In B-Human logs, the time is offset by 100_000 ms
            return time - 100_000

    def image(self) -> np.ndarray | None:
        """Returns the BGR image of the frame, if available.

        :return: The BGR image of the frame, if available.
        """
        if (jpeg_image := self.get("JPEGImage")) is not None:
            timestamp = jpeg_image.get("timestamp")
            assert timestamp is not None, "Timestamp must be defined for JPEGImage"

            size = jpeg_image.get("size")
            assert size is not None, "Size must be defined for JPEGImage"

            height = jpeg_image.get("height")
            assert height is not None, "Height must be defined for JPEGImage"

            width = jpeg_image.get("width")
            assert width is not None, "Width must be defined for JPEGImage"

            data = jpeg_image.get("_data")
            assert data is not None, "Data must be defined for JPEGImage"
            data = data[-size:]

            # Load YUYV data
            img = Image.open(io.BytesIO(data))
            img_yuyv = np.array(img)

            # Convert YUYV to YUV format (2 channels per pixel)
            # Prepare empty YUV image
            img_yuv = np.empty((height * 2, width * 2, 3), dtype=np.uint8)
            # Extract Y, U, V channels
            y0 = img_yuyv[:, :, 0]
            u = img_yuyv[:, :, 1]
            y1 = img_yuyv[:, :, 2]
            v = img_yuyv[:, :, 3]
            # Fill YUV image with Y, U, V channels
            img_yuv[:, ::2, 0] = y0
            img_yuv[:, 1::2, 0] = y1
            img_yuv[:, ::2, 1] = u
            img_yuv[:, 1::2, 1] = u
            img_yuv[:, ::2, 2] = v
            img_yuv[:, 1::2, 2] = v

            # Convert YUV to BGR
            img_bgr = 255 - cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return img_bgr


class BHumanImportStrategy(ImportStrategy):
    def __init__(
        self,
        metadata: ImportMetadata,
        image_converter: ImageConverter,
        game_state_converter: GameStateConverter,
        synced_data_converter: SyncedDataConverter,
        caching: bool = False,
        video: bool = False,
    ):
        self.metadata = metadata
        self.image_converter = image_converter
        self.game_state_converter = game_state_converter
        self.synced_data_converter = synced_data_converter
        self.caching = caching
        self.video = video

        self.datetime: datetime | None = None

        self.model_data = ModelData()

    def convert_to_model_data(self, file_path: Path) -> ModelData:
        self.verify_file(file_path)
        self.datetime = self.get_datetime_from_file_path(file_path)

        self.model_data.recording = self._create_recording(file_path)
        log, frames = self._read_log_file(file_path)
        self._populate_timestamps(frames)

        # TODO: populate team_color and img_width_scaling, img_height_scaling

        for frame in frames:
            self._show_video(frame)

        return self.model_data

    def verify_file(self, file_path: Path) -> bool:
        # Check file prefix .log
        if file_path.suffix != ".log":
            logger.error("File is not a .log file. Exiting.")
            sys.exit(1)

        # bhumand_ prefix is used for text logs
        if "bhumand_" in file_path.name:
            logger.error("File is just a text log, not a B-Human log file. Exiting.")
            sys.exit(1)

        return True

    def get_datetime_from_file_path(self, file_path: Path) -> datetime:
        """Extracts the datetime from the file path.
        We assume the date is part of the file path.
        Example: */<DATE>/<GAME>/<ROBOT>/<LOG_FILE>.log
        <DATE> can be in the formats:
        - *
        - YYYY-MM-DD
        - YYYY_MM_DD
        - YYYY-MM-DD*
        - YYYY_MM_DD*
        - YYYY-MM-DD_HH-MM*

        <GAME> can also include more detailed datetime and can be:
        - *
        - YYYY-MM-DD-HH-MM


        :param file_path: Path to the file
        :return: Extracted datetime
        """
        # Extract date from <GAME> part of the file path
        game = file_path.parent.name
        # Cut off alphabetical characters and -_ at the end of the string
        game = game.rstrip("abcdefghijklmnopqrstuvwxyzäöüABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ-_")
        try:
            parsed = dateutil_parse(game, fuzzy=True, ignoretz=True)
            logger.debug(f"Extracted datetime from game part of the file path: {parsed}")
            return parsed
        except ParserError:
            pass

        # Extract date from <DATE> part of the file path
        date = file_path.parent.parent.name
        # Cut off alphabetical characters and -_ at the end of the string
        date = date.rstrip("abcdefghijklmnopqrstuvwxyzäöüABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ-_")
        try:
            parsed = dateutil_parse(date, fuzzy=True, ignoretz=True)
            logger.debug(f"Extracted datetime from date part of the file path: {parsed}")
            return parsed
        except ParserError:
            pass

        logger.error(f"Could not extract datetime from file path: {file_path}")
        sys.exit(1)

    def _read_log_file(self, file_path: Path) -> tuple[Log, list[SmartFrame]]:
        log = Log(str(file_path), keep_going=True)

        cache_file = Path("/tmp") / Path(file_path.name).with_suffix(".pkl")
        if self.caching and cache_file.exists():
            import pickle

            logger.info(f"Reading B-Human data from cached file '{cache_file}'...")
            with open(cache_file, "rb") as file:
                frames = pickle.load(file)
            return log, frames

        logger.debug(f"Reading B-Human data from file '{file_path}'...")
        # Read and pre-process frames
        frames: list[SmartFrame] = [
            SmartFrame.from_frame(frame)
            for frame in tqdm(
                log,
                desc="Reading frames",
                unit="frames",
                total=len(log),
            )
        ]

        logger.debug(
            f"Read log with {len(log)} Frames: {log.bodyName=}, {log.headName=}, {log.identifier=}, {log.location=}, "
            f"{log.playerNumber=}, {log.scenario=}, {log.suffix=}"
        )

        if self.caching:
            # Pickle dump frames
            import pickle

            with open(cache_file, "wb") as file:
                pickle.dump(frames, file)
        return log, frames

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
            img_width_scaling=0.0,  # Needs to be overwritten when processing images
            img_height_scaling=0.0,  # Needs to be overwritten when processing images
        )
        return recording

    def _extract_timeframe(self, frames: list[SmartFrame]) -> tuple[timedelta, timedelta]:
        first_time_ms: int | None = None
        for frame in frames:
            if (time_ms := frame.time_ms()) is not None:
                first_time_ms = time_ms
                break
        assert isinstance(first_time_ms, int), "Timestamps (ms) must be integers"
        logger.debug(f"Timestamp offset: {first_time_ms} [ms]")

        last_time_ms: int | None = None
        for frame in reversed(frames):
            if (time_ms := frame.time_ms()) is not None:
                last_time_ms = time_ms
                break
        assert isinstance(last_time_ms, int), "Timestamps (ms) must be integers"

        assert first_time_ms <= last_time_ms, "First timestamp must be before last timestamp"

        return timedelta(milliseconds=first_time_ms), timedelta(milliseconds=last_time_ms)

    def _populate_timestamps(self, frames: list[SmartFrame]) -> None:
        first_timestamp, last_timestamp = self._extract_timeframe(frames)

        assert self.datetime is not None, "Datetime must be defined to populate timestamps"
        start_time: datetime = self.datetime + first_timestamp
        end_time: datetime = self.datetime + last_timestamp
        assert start_time <= end_time, "Start time must be before end time"

        assert self.model_data.recording is not None, "Recording must be defined to populate timestamps"
        self.model_data.recording.start_time = start_time
        self.model_data.recording.end_time = end_time

        logger.info(
            f"Recording duration {self.model_data.recording.duration().total_seconds()} [s]"
            f" from {start_time.isoformat()}"
        )

    def _show_video(self, frame: SmartFrame) -> None:
        if self.video and (img := frame.image()) is not None:
            cv2.imshow(frame.thread, img)
            cv2.waitKey(1)
