from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from mcap.reader import McapReader, make_reader
from mcap.summary import Summary
from mcap_ros2.decoder import DecoderFactory

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.db import Database
from ddlitlab2024.dataset.models import GameState, JointCommands, JointStates, Recording, RobotState, TeamColor
from ddlitlab2024.utils.utils import camelcase_to_snakecase

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


class RosBagToModelMapper:
    def __init__(self, rosbag_path: Path, db: Database):
        self.bag_path: Path = rosbag_path
        self.db: Database = db
        self.first_used_message_time: int | None = None
        self.models = {}

    def read(self):
        with open(self.bag_path, "rb") as f:
            reader: McapReader = make_reader(f, decoder_factories=[DecoderFactory()])
            summary: Summary | None = reader.get_summary()

            if summary is None:
                logger.error("No summary found in the MCAP file, skipping processing.")
                return

            recording: Recording = self.create_recording(summary)
            self.models = {
                "recording": recording,
                "game_states": [],
                "joint_states": [],
                "joint_commands": [],
            }
            self._log_debug_info(summary, recording)

            for _, channel, message, ros_msg in reader.iter_decoded_messages(topics=USED_TOPICS):
                self.first_used_message_time = self.first_used_message_time or message.publish_time
                assert self.first_used_message_time is not None, "First used message, did not have a publish time"

                relative_timestamp = self.calculate_relative_timestamp(message.publish_time)

                match channel.topic:
                    case "/gamestate":
                        recording.team_color = TeamColor.BLUE if ros_msg.team_color == 0 else TeamColor.RED
                        self.models.get("game_states").append(
                            self.create_gamestate(ros_msg, relative_timestamp, recording)
                        )
                    case "/joint_states":
                        self.models.get("joint_states").append(
                            self.create_joint_states(ros_msg, relative_timestamp, recording)
                        )
                    case "/DynamixelController/command":
                        self.models.get("joint_commands").append(
                            self.create_joint_commands(ros_msg, relative_timestamp, recording)
                        )

        self.db.session.add_all(
            self.models.get("game_states") + self.models.get("joint_states") + self.models.get("joint_commands")
        )
        self.db.session.commit()

    def create_recording(self, summary: Summary):
        start_timestamp, end_timestamp = self.extract_timeframe(summary)

        return Recording(
            allow_public=True,
            original_file=self.bag_path.name,
            team_name="Bit-Bots",
            robot_type="Wolfgang-OP",
            start_time=datetime.fromtimestamp(start_timestamp / 1e9),
            end_time=datetime.fromtimestamp(end_timestamp / 1e9),
            location="RoboCup 2024",
            simulated=False,
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
        joint_states_dict = {
            camelcase_to_snakecase(name): self.shift_radian_to_positive_range(position)
            for name, position in joint_states_data
        }

        return JointStates(stamp=relative_timestamp, recording=recording, **joint_states_dict)

    def create_joint_commands(self, msg, relative_timestamp: float, recording: Recording) -> JointStates:
        joint_commands_data = list(zip(msg.joint_names, msg.positions))

        return JointCommands(
            stamp=relative_timestamp, recording=recording, **self.joint_commands_dict_from_msg_data(joint_commands_data)
        )

    def joint_commands_dict_from_msg_data(self, joint_commands_data: list[tuple[str, float]]) -> dict[str, float]:
        joint_commands_dict = {}

        for name, position in joint_commands_data:
            key = camelcase_to_snakecase(name)
            value = self.shift_radian_to_positive_range(position)
            joint_commands_dict[key] = value

        return joint_commands_dict

    def shift_radian_to_positive_range(self, principal_range_radian: float) -> float:
        """
        Shift the principal range radian [-pi, pi] to the positive principal range [0, 2pi].
        """
        return (principal_range_radian + 2 * np.pi) % (2 * np.pi)

    def calculate_relative_timestamp(self, publish_time: int) -> float:
        """
        Calculate the relative timestamp offset in seconds (float) from the first used message publish time.
        """
        assert self.first_used_message_time is not None, "First used message, did not have a publish time"
        return (publish_time - self.first_used_message_time) / 1e9

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
