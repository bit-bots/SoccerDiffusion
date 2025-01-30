import io
import re
import sys
from collections.abc import MutableMapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypeAlias, TypeVar

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pybh.logs import Array, Frame, Log, Record, Value
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

global GLOBAL_TIME_OFFSET
global JPEG_IMAGE_DATE_OFFSET

GLOBAL_TIME_OFFSET: int | None = None
JPEG_IMAGE_DATE_OFFSET: int | None = None

SmartValue: TypeAlias = bool | int | float | str | bytes | list | dict | Value


class SmartRecord(MutableMapping):
    def __init__(self, record: Record) -> None:
        self.data: dict[str, SmartValue] = {}
        for key in record:
            value = record.__getattr__(key)
            match value:
                case Record():
                    self.data[key] = SmartRecord(value).data
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

    def __getitem__(self, key: str) -> SmartValue:
        return self.data[key]

    def __setitem__(self, key: str, value: SmartValue) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    # Typedef for default value in get method
    D = TypeVar("D")

    def get(self, key: str, default: D = None) -> SmartValue | D:
        return self.data.get(key, default)


class SmartFrame(MutableMapping):
    def __init__(self, frame: Frame):
        self.data: dict[str, SmartRecord] = {
            repr: SmartRecord(frame[repr]) for repr in frame.representations if repr in USED_REPRESENTATIONS
        }
        self.thread: str = frame.thread

    def __getitem__(self, key: str) -> SmartRecord:
        return self.data[key]

    def __setitem__(self, key: str, value: SmartRecord) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def get(self, key, default=None) -> SmartRecord | None:
        return self.data.get(key, default)

    @staticmethod
    def from_frame(frame: Frame) -> "SmartFrame":
        return SmartFrame(frame)

    def time_ms(self) -> int | None:
        """Returns the zero-shifted time of the frame in milliseconds.

        :return: The zero-shifted time of the frame in milliseconds, if available
        """
        global GLOBAL_TIME_OFFSET, JPEG_IMAGE_DATE_OFFSET

        # Case JPEGImage timestamp is available
        # Shift the time using the GLOBAL_TIME_OFFSET and JPEGImage_DATE_OFFSET if they are defined
        if (image := self.get("JPEGImage")) is not None and isinstance((timestamp := image.get("timestamp")), int):
            if GLOBAL_TIME_OFFSET is not None and JPEG_IMAGE_DATE_OFFSET is not None:
                return timestamp - GLOBAL_TIME_OFFSET - JPEG_IMAGE_DATE_OFFSET

        # Case other timestamp is available
        # Shift the time using the GLOBAL_TIME_OFFSET if it is defined
        if isinstance((time := self.get("time")), int):
            if GLOBAL_TIME_OFFSET is not None:
                return time - GLOBAL_TIME_OFFSET

        # If no time is available, return None
        return None

    def raw_time_ms_meta(self) -> list[tuple[int, str, str]]:
        """Returns a list of raw times contained in the frame in milliseconds since the start of the log.
        Meta data is the thread name of the frame and the representation of the frame.

        :return: A list of (time, thread, representation) tuples.
        """
        times = []
        for representation, record in self.items():
            if (time := record.get("time")) is not None:
                times.append((time, self.thread, representation))
            if (timestamp := record.get("timestamp")) is not None:
                times.append((timestamp, self.thread, representation))
        return times

    def image(self) -> np.ndarray | None:
        """Returns the BGR image of the frame, if available.

        :return: The BGR image of the frame, if available.
        """
        if (jpeg_image := self.get("JPEGImage")) is not None:
            timestamp: int = jpeg_image.get("timestamp")  # type: ignore
            assert timestamp is not None, "Timestamp must be defined for JPEGImage"

            size: int = jpeg_image.get("size")  # type: ignore
            assert size is not None, "Size must be defined for JPEGImage"

            height: int = jpeg_image.get("height")  # type: ignore
            assert height is not None, "Height must be defined for JPEGImage"

            width: int = jpeg_image.get("width")  # type: ignore
            assert width is not None, "Width must be defined for JPEGImage"

            data: bytes = jpeg_image.get("_data")  # type: ignore
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
        frames = self.handle_timestamps(frames)

        # TODO: populate team_color and img_width_scaling, img_height_scaling

        # # Scatter plot the time of sorted frames
        # import matplotlib.pyplot as plt

        # times = [frame.time_ms() for frame in frames]
        # plt.scatter(range(len(times)), times)  # type: ignore
        # plt.xlabel("Frame Index")
        # plt.ylabel("Time [ms]")
        # plt.title("Time of Sorted Frames")
        # plt.show()

        for i, frame in enumerate(frames):
            self._show_video(frame)

        sys.exit(0)
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
        The following formats have been observed:
        - YYYY-MM-DD
        - YYYY_MM_DD
        - YYYY-MM-DD*
        - YYYY_MM_DD*
        - YYYY-MM-DD_HH-MM*

        :param file_path: Path to the file
        :return: Extracted datetime
        """
        # Cut off left parts of the path, only 4 parts are considered
        path = Path.joinpath(Path(), *file_path.parts[-5:-1])

        pattern: str = (
            r"20(\d{2})[-_.:\s](\d{1,2})[-_.:\s](\d{1,2})"  # Match the year, month, and day
            r"(?:[-_.:\s]+(\d{1,2})[-_.:\s](\d{1,2}))?"  # Match the time if present, separated by one or more delimiters
        )
        # Find all matches in the part
        matches = re.findall(pattern, str(path))

        # Keep track of the longest match (time-inclusive if possible)
        longest_match = None
        for match in matches:
            year_suffix = int(match[0]) + 2000  # Convert year to full (e.g., 23 -> 2023)
            month = int(match[1])
            day = int(match[2])
            hour = int(match[3]) if match[3] else 0  # Default to 00:00 if no time is present
            minute = int(match[4]) if match[4] else 0

            # Create a datetime object for the match
            current_datetime = datetime(year=year_suffix, month=month, day=day, hour=hour, minute=minute)

            # Compare current match to the longest_match
            if longest_match is None or (match[3] and match[4]):  # Prioritize matches with time
                longest_match = current_datetime

        if longest_match is not None:
            return longest_match

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

    def handle_timestamps(self, frames: list[SmartFrame]) -> list[SmartFrame]:
        """Shifts the timestamps of the frames to start at 0 ms and populates the recording start and end times.

        :param frames: List of frames
        :return: List of sorted frames with shifted timestamps (Drops frames without timestamps)
        """

        global GLOBAL_TIME_OFFSET, JPEG_IMAGE_DATE_OFFSET

        times: list[tuple[int, int, str, str]] = []  # Frame index, time, thread, representation
        for i, frame in enumerate(frames):
            for time, thread, representation in frame.raw_time_ms_meta():
                times.append((i, time, thread, representation))
        df_times = pd.DataFrame(times, columns=["Frame Index", "Time [ms]", "Thread", "Representation"])

        # Timestamps of JPEGImage frames are offset to everything else by about 25 days.
        # Therefore, we need to subtract the offset from the timestamps of JPEGImage frames.
        # We assume that the offset is constant for all JPEGImage frames.
        # We calculate the offset by taking the average time of all JPEGImage frames and
        # subtracting the average time of all frames.
        avg_times = df_times.groupby("Representation")["Time [ms]"].mean().reset_index()
        JPEG_IMAGE_DATE_OFFSET = int(
            avg_times[avg_times["Representation"] == "JPEGImage"]["Time [ms]"].values[0]
            - avg_times[avg_times["Representation"] != "JPEGImage"]["Time [ms]"].mean()
        )

        # Subtract the offset from the timestamps of JPEGImage frames
        df_times.loc[df_times["Representation"] == "JPEGImage", "Time [ms]"] -= JPEG_IMAGE_DATE_OFFSET

        # Shift all timestamps to start at 0 ms
        # Get the minimum timestamp of all frames and subtract it from all timestamps
        GLOBAL_TIME_OFFSET = int(df_times["Time [ms]"].min())
        df_times["Time [ms]"] -= GLOBAL_TIME_OFFSET

        first_timestamp: timedelta = timedelta(milliseconds=int(df_times["Time [ms]"].min()))
        last_timestamp: timedelta = timedelta(milliseconds=int(df_times["Time [ms]"].max()))

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

        # Drop frames with time_ms() == None and sort frames by time_ms() in ascending order
        frames = [frame for frame in frames if frame.time_ms() is not None]
        frames.sort(key=lambda frame: frame.time_ms())  # type: ignore
        return frames

    def _show_video(self, frame: SmartFrame) -> None:
        if self.video and (img := frame.image()) is not None:
            cv2.imshow(frame.thread, img)
            cv2.waitKey(1)
