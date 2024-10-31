import shutil
from pathlib import Path

import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, JointState
from sqlalchemy.orm import Session
from std_msgs.msg import Header, String

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.models import Recording, stamp_to_seconds_nanoseconds


def get_recording(db: Session, recording_id_or_filename: str | int) -> Recording:
    """Get the recording from the input string or integer

    param db: The database
    param recording_id_or_filename: The recording ID or original filename
    raises ValueError: If the recording does not exist
    return: The recording
    """
    if isinstance(recording_id_or_filename, int) or recording_id_or_filename.isdigit():
        # Verify that the recording exists
        recording_id = int(recording_id_or_filename)
        recording = db.query(Recording).get(recording_id)
        if recording is None:
            raise ValueError(f"Recording '{recording_id}' not found")
        return recording
    elif isinstance(recording_id_or_filename, str):
        recording = db.query(Recording).filter(Recording.original_file == recording_id_or_filename).first()
        if recording is None:
            raise ValueError(f"Recording with original filename '{recording_id_or_filename}' not found")
        return recording
    else:
        raise TypeError("Recording ID must be an integer or string")


def get_writer(output: Path) -> rosbag2_py.SequentialWriter:
    """Get the mcap writer.

    param output: The output mcap file
    return: The mcap writer
    """
    if output.exists():
        # Remove the existing directory
        shutil.rmtree(output)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return writer


def write_images(recording: Recording, writer: rosbag2_py.SequentialWriter) -> None:
    """Write the images to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/image", type="sensor_msgs/msg/Image", serialization_format="cdr")
    )

    # Write images
    logger.info("Writing images")
    for image in recording.images:
        seconds, nanoseconds = stamp_to_seconds_nanoseconds(image.stamp)
        image_msg = Image(
            header=Header(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id="camera_optical"),
            height=recording.img_height,
            width=recording.img_width,
            encoding="rgb8",
            is_bigendian=0,
            step=recording.img_width * 3,
            data=image.data,
        )
        writer.write("/image", serialize_message(image_msg), int(image.stamp * 1_000_000_000))


def write_rotations(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the rotations to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/rotation", type="geometry_msgs/msg/Quaternion", serialization_format="cdr")
    )

    # Write rotations
    logger.info("Writing rotations")
    for rotation in recording.rotations:
        rotation_msg = Quaternion(
            x=rotation.x,
            y=rotation.y,
            z=rotation.z,
            w=rotation.w,
        )
        writer.write("/rotation", serialize_message(rotation_msg), int(rotation.stamp * 1_000_000_000))


def write_joint_states(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the joint states to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/joint_states", type="sensor_msgs/msg/JointState", serialization_format="cdr")
    )

    # Write joint states
    logger.info("Writing joint states")
    for joint_state in recording.joint_states:
        seconds, nanoseconds = stamp_to_seconds_nanoseconds(joint_state.stamp)
        joints: list[tuple[str, float]] = [
            ("r_shoulder_pitch", joint_state.r_shoulder_pitch),
            ("l_shoulder_pitch", joint_state.l_shoulder_pitch),
            ("r_shoulder_roll", joint_state.r_shoulder_roll),
            ("l_shoulder_roll", joint_state.l_shoulder_roll),
            ("r_elbow", joint_state.r_elbow),
            ("l_elbow", joint_state.l_elbow),
            ("r_hip_yaw", joint_state.r_hip_yaw),
            ("l_hip_yaw", joint_state.l_hip_yaw),
            ("r_hip_roll", joint_state.r_hip_roll),
            ("l_hip_roll", joint_state.l_hip_roll),
            ("r_hip_pitch", joint_state.r_hip_pitch),
            ("l_hip_pitch", joint_state.l_hip_pitch),
            ("r_knee", joint_state.r_knee),
            ("l_knee", joint_state.l_knee),
            ("r_ankle_pitch", joint_state.r_ankle_pitch),
            ("l_ankle_pitch", joint_state.l_ankle_pitch),
            ("r_ankle_roll", joint_state.r_ankle_roll),
            ("l_ankle_roll", joint_state.l_ankle_roll),
            ("head_pan", joint_state.head_pan),
            ("head_tilt", joint_state.head_tilt),
        ]
        joint_state_msg = JointState(
            header=Header(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id="base_link"),
            name=[name for name, _ in joints],
            position=[position for _, position in joints],
            velocity=[0.0] * len(joints),
            effort=[0.0] * len(joints),
        )
        writer.write("/joint_states", serialize_message(joint_state_msg), int(joint_state.stamp * 1_000_000_000))


def write_joint_commands(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the joint commands to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/joint_commands", type="sensor_msgs/msg/JointState", serialization_format="cdr")
    )

    # Write joint commands
    logger.info("Writing joint commands")
    for joint_command in recording.joint_commands:
        seconds, nanoseconds = stamp_to_seconds_nanoseconds(joint_command.stamp)
        joints: list[tuple[str, float]] = [
            ("r_shoulder_pitch", joint_command.r_shoulder_pitch),
            ("l_shoulder_pitch", joint_command.l_shoulder_pitch),
            ("r_shoulder_roll", joint_command.r_shoulder_roll),
            ("l_shoulder_roll", joint_command.l_shoulder_roll),
            ("r_elbow", joint_command.r_elbow),
            ("l_elbow", joint_command.l_elbow),
            ("r_hip_yaw", joint_command.r_hip_yaw),
            ("l_hip_yaw", joint_command.l_hip_yaw),
            ("r_hip_roll", joint_command.r_hip_roll),
            ("l_hip_roll", joint_command.l_hip_roll),
            ("r_hip_pitch", joint_command.r_hip_pitch),
            ("l_hip_pitch", joint_command.l_hip_pitch),
            ("r_knee", joint_command.r_knee),
            ("l_knee", joint_command.l_knee),
            ("r_ankle_pitch", joint_command.r_ankle_pitch),
            ("l_ankle_pitch", joint_command.l_ankle_pitch),
            ("r_ankle_roll", joint_command.r_ankle_roll),
            ("l_ankle_roll", joint_command.l_ankle_roll),
            ("head_pan", joint_command.head_pan),
            ("head_tilt", joint_command.head_tilt),
        ]
        joint_command_msg = JointState(
            header=Header(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id="base_link"),
            name=[name for name, _ in joints],
            position=[position for _, position in joints],
            velocity=[0.0] * len(joints),
            effort=[0.0] * len(joints),
        )
        writer.write("/joint_commands", serialize_message(joint_command_msg), int(joint_command.stamp * 1_000_000_000))


def write_game_states(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the game states to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/game_state", type="std_msgs/msg/String", serialization_format="cdr")
    )

    # Write game states
    logger.info("Writing game states")
    for game_state in recording.game_states:
        game_state_msg = String(data=game_state.state)
        writer.write("/game_state", serialize_message(game_state_msg), int(game_state.stamp * 1_000_000_000))


def recording2mcap(db: Session, recording_id_or_filename: str | int, output: Path) -> None:
    """Convert a recording to an mcap file

    param db: The database
    param recording_id_or_filename: The recording ID or original filename
    param output: The output mcap file
    """
    recording = get_recording(db, recording_id_or_filename)
    logger.info(f"Converting recording '{recording._id}' to mcap file '{output}'")

    writer = get_writer(output)
    write_images(recording, writer)
    write_rotations(recording, writer)
    write_joint_states(recording, writer)
    write_joint_commands(recording, writer)
    write_game_states(recording, writer)
    del writer

    logger.info(f"Recording '{recording._id}' converted to mcap file '{output}'")
