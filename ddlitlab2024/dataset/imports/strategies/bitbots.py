from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap.summary import Summary
from mcap_ros2.decoder import DecoderFactory

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.imports.model_importer import ImportMetadata, ImportStrategy, ModelData
from ddlitlab2024.dataset.models import (
    DEFAULT_IMG_SIZE,
    GameState,
    Image,
    JointCommands,
    JointStates,
    Recording,
    RobotState,
    TeamColor,
)
from ddlitlab2024.utils.utils import camelcase_to_snakecase, shift_radian_to_positive_range

DATETIME_FORMAT = "%d.%m-%Y %H:%M:%S"


class GameStateMessage(Enum):
    INITIAL = 0
    READY = 1
    SET = 2
    PLAYING = 3
    FINISHED = 4


TOPIC_LIST_GO = [
    "/DynamixelController/command",
    "/animation",
    "/audio/audio",
    "/ball_obstacle_active",
    "/ball_position_relative_filtered",
    "/ball_relative_filtered",
    "/ball_relative_movement",
    "/balls_relative",
    "/camera/camera_info",
    "/camera/image_to_record",
    "/cmd_vel",
    "/debug/approach_point",
    "/debug/ball_twist",
    "/debug/dsd/body_behavior/dsd_current_action",
    "/debug/dsd/body_behavior/dsd_stack",
    "/debug/dsd/body_behavior/dsd_tree",
    "/debug/dsd/hcm/dsd_current_action",
    "/debug/dsd/hcm/dsd_stack",
    "/debug/dsd/hcm/dsd_tree",
    "/debug/dsd/localization/dsd_current_action",
    "/debug/dsd/localization/dsd_stack",
    "/debug/dsd/localization/dsd_tree",
    "/debug/used_ball",
    "/debug/which_ball_is_used",
    "/diagnostics",
    "/diagnostics_agg",
    "/field/map",
    "/gamestate",
    "/goal_pose",
    "/head_mode",
    "/motion_odometry",
    "/pose_with_covariance",
    "/robot_state",
    "/robots_relative",
    "/robots_relative_filtered",
    "/rosout",
    "/strategy",
    "/system_workload",
    "/team_data",
    "/tf",
    "/tf_static",
    "/time_to_ball",
]

TOPIC_LIST_RBC24 = [
    "/DynamixelController/command",
    "/audio/audio",
    "/ball_position_relative_filtered",
    "/balls_relative",
    "/camera/camera_info",
    "/camera/image_proc",
    "/cmd_vel",
    "/core/power_switch_status",
    "/debug/dsd/body_behavior/dsd_stack",
    "/debug/dsd/body_behavior/dsd_tree",
    "/debug/dsd/hcm/dsd_current_action",
    "/debug/dsd/hcm/dsd_stack",
    "/debug/dsd/hcm/dsd_tree",
    "/debug/dsd/localization/dsd_stack",
    "/debug/dsd/localization/dsd_tree",
    "/diagnostics",
    "/diagnostics_agg",
    "/field/map",
    "/joint_states",
    "/motion_odometry",
    "/pose_with_covariance",
    "/robot_state",
    "/robots_relative",
    "/robots_relative_filtered",
    "/rosout",
    "/strategy",
    "/system_workload",
    "/tf",
    "/tf_static",
    "/time_to_ball",
    "/workspace_status",
]

USED_TOPICS = [
    "/DynamixelController/command",
    "/camera/camera_info",
    "/camera/image_proc",
    "/gamestate",
    "/imu/data",
    "/joint_states",
    "/tf",
]


class BitBotsImportStrategy(ImportStrategy):
    def __init__(self, metadata: ImportMetadata):
        self.metadata = metadata

    def convert_to_model_data(self, file_path: Path) -> ModelData:
        with self._mcap_reader(file_path) as reader:
            summary: Summary | None = reader.get_summary()

            if summary is None:
                logger.error("No summary found in the MCAP file, skipping processing.")
                return ModelData()

            first_used_msg_time = None
            model_data = ModelData(recording=self.create_recording(summary, file_path))
            assert model_data.recording is not None, "Recording is not set"

            self._log_debug_info(summary, model_data.recording)

            for _, channel, message, ros_msg in reader.iter_decoded_messages(topics=USED_TOPICS):
                first_used_msg_time = first_used_msg_time or message.publish_time
                relative_timestamp = (message.publish_time - first_used_msg_time) / 1e9

                match channel.topic:
                    case "/gamestate":
                        model_data.recording.team_color = TeamColor.BLUE if ros_msg.team_color == 0 else TeamColor.RED
                        model_data.game_states.append(
                            self.create_gamestate(ros_msg, relative_timestamp, model_data.recording)
                        )
                    case "/joint_states":
                        model_data.joint_states.append(
                            self.create_joint_states(ros_msg, relative_timestamp, model_data.recording)
                        )
                    case "/DynamixelController/command":
                        model_data.joint_commands.append(
                            self.create_joint_commands(ros_msg, relative_timestamp, model_data.recording)
                        )
                    case "/camera/image_proc" | "/camera/image_raw":
                        model_data.recording.img_width_scaling = DEFAULT_IMG_SIZE[0] / ros_msg.width
                        model_data.recording.img_height_scaling = DEFAULT_IMG_SIZE[1] / ros_msg.height
                        model_data.images.append(self.create_image(ros_msg, relative_timestamp, model_data.recording))

        return model_data

    def create_image(self, msg, relative_timestamp: float, recording: Recording) -> Image:
        img_array = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 3))

        will_img_be_upscaled = recording.img_width_scaling > 1.0 or recording.img_height_scaling > 1.0
        interpolation = cv2.INTER_AREA
        if will_img_be_upscaled:
            interpolation = cv2.INTER_CUBIC

        resized_img = cv2.resize(img_array, (recording.img_width, recording.img_height), interpolation=interpolation)
        resized_rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        return Image(
            stamp=relative_timestamp,
            recording=recording,
            image=resized_rgb_img,
        )

    def create_recording(self, summary: Summary, mcap_file_path: Path) -> Recording:
        start_timestamp, end_timestamp = self.extract_timeframe(summary)

        return Recording(
            allow_public=self.metadata.allow_public,
            original_file=mcap_file_path.name,
            team_name=self.metadata.team_name,
            robot_type=self.metadata.robot_type,
            start_time=datetime.fromtimestamp(start_timestamp / 1e9),
            end_time=datetime.fromtimestamp(end_timestamp / 1e9),
            location=self.metadata.location,
            simulated=self.metadata.simulated,
            img_width=DEFAULT_IMG_SIZE[0],
            img_height=DEFAULT_IMG_SIZE[1],
            # needs to be overwritten when processing images
            img_width_scaling=0.0,
            img_height_scaling=0.0,
        )

    def create_gamestate(self, msg, relative_timestamp: float, recording: Recording) -> GameState:
        return GameState(stamp=relative_timestamp, recording=recording, state=self.robot_state_from_msg(msg))

    def robot_state_from_msg(self, msg) -> RobotState:
        if msg.penalized:
            return RobotState.STOPPED

        match msg.game_state:
            case GameStateMessage.INITIAL:
                return RobotState.STOPPED
            case GameStateMessage.READY:
                return RobotState.POSITIONING
            case GameStateMessage.SET:
                return RobotState.STOPPED
            case GameStateMessage.PLAYING:
                return RobotState.PLAYING
            case GameStateMessage.FINISHED:
                return RobotState.STOPPED
            case _:
                return RobotState.UNKNOWN

    def create_joint_states(self, msg, relative_timestamp: float, recording: Recording) -> JointStates:
        joint_states_data = list(zip(msg.name, msg.position))

        return JointStates(
            stamp=relative_timestamp, recording=recording, **self._joints_dict_from_msg_data(joint_states_data)
        )

    def create_joint_commands(self, msg, relative_timestamp: float, recording: Recording) -> JointStates:
        joint_commands_data = list(zip(msg.joint_names, msg.positions))

        return JointCommands(
            stamp=relative_timestamp, recording=recording, **self._joints_dict_from_msg_data(joint_commands_data)
        )

    def _joints_dict_from_msg_data(self, joints_data: list[tuple[str, float]]) -> dict[str, float]:
        joints_dict = {}

        for name, position in joints_data:
            key = camelcase_to_snakecase(name)
            value = shift_radian_to_positive_range(position)
            joints_dict[key] = value

        return joints_dict

    def extract_timeframe(self, summary: Summary) -> tuple[int, int]:
        first_msg_start_time = None
        last_msg_end_time = None

        for chunk_index in summary.chunk_indexes:
            if first_msg_start_time is None or chunk_index.message_start_time < first_msg_start_time:
                first_msg_start_time = chunk_index.message_start_time
            if last_msg_end_time is None or chunk_index.message_end_time > last_msg_end_time:
                last_msg_end_time = chunk_index.message_end_time

        assert first_msg_start_time is not None, "No start time found in the MCAP file"
        assert last_msg_end_time is not None, "No end time found in the MCAP file"

        return first_msg_start_time, last_msg_end_time

    @contextmanager
    def _mcap_reader(self, mcap_file_path: Path):
        with open(mcap_file_path, "rb") as f:
            yield make_reader(f, decoder_factories=[DecoderFactory()])

    def _log_debug_info(self, summary: Summary, recording: Recording):
        log_message = f"Processing rosbag: {recording.original_file} - {recording.team_name}"
        if recording.location:
            log_message += f" {recording.location}"
        if recording.start_time:
            log_message += f": {recording.start_time}"

        available_topics = [channel.topic for channel in summary.channels.values()]
        log_message += f"\nAvailable topics: {available_topics}"
        log_message += f"\nUsed topics: {USED_TOPICS}"

        logger.info(log_message)
