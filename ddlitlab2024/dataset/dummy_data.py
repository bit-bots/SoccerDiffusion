import datetime
import math
import random

import numpy as np
from sqlalchemy.orm import Session

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.models import (
    GameState,
    Image,
    JointCommand,
    JointState,
    Recording,
    RobotState,
    Rotation,
    TeamColor,
)


def insert_recordings(db: Session, n) -> list[int]:
    logger.debug("Inserting recordings...")
    for i in range(n):
        db.add(
            Recording(
                allow_public=True,
                original_file=f"dummy_original_file{i}",
                team_name=f"dummy_team_name{i}",
                team_color=random.choice(list(TeamColor)),
                robot_type=f"dummy_robot_type{i}",
                start_time=datetime.datetime.now(),
                location=f"dummy_location{i}",
                simulated=True,
                img_width_scaling=1.0,
                img_height_scaling=1.0,
            ),
        )
    db.flush()  # Ensure the recording is written to the database and the ID is generated
    recording = db.query(Recording).order_by(Recording._id.desc()).limit(n).all()
    if recording is None:
        raise ValueError("Failed to insert recordings")
    return [r._id for r in reversed(recording)]


def insert_images(db: Session, recording_ids: list[int], n: int) -> None:
    for recording_id in recording_ids:
        # Get width and height from the recording
        recording = db.query(Recording).get(recording_id)
        if recording is None:
            raise ValueError(f"Recording '{recording_id}' not found")
        for i in range(n):
            db.add(
                Image(
                    stamp=float(i),
                    recording_id=recording_id,
                    data=np.random.randint(
                        0, 255, (recording.img_height, recording.img_width, 3), dtype=np.uint8
                    ).tobytes(),
                )
            )


def insert_rotations(db: Session, recording_ids: list[int], n: int) -> None:
    for recording_id in recording_ids:
        for i in range(n):
            db.add(
                Rotation(
                    stamp=float(i),
                    recording_id=recording_id,
                    x=random.random(),
                    y=random.random(),
                    z=random.random(),
                    w=random.random(),
                ),
            )


def insert_joint_states(db: Session, recording_ids: list[int], n: int) -> None:
    for recording_id in recording_ids:
        for i in range(n):
            db.add(
                JointState(
                    stamp=float(i),
                    recording_id=recording_id,
                    r_shoulder_pitch=random.random() * 2 * math.pi,
                    l_shoulder_pitch=random.random() * 2 * math.pi,
                    r_shoulder_roll=random.random() * 2 * math.pi,
                    l_shoulder_roll=random.random() * 2 * math.pi,
                    r_elbow=random.random() * 2 * math.pi,
                    l_elbow=random.random() * 2 * math.pi,
                    r_hip_yaw=random.random() * 2 * math.pi,
                    l_hip_yaw=random.random() * 2 * math.pi,
                    r_hip_roll=random.random() * 2 * math.pi,
                    l_hip_roll=random.random() * 2 * math.pi,
                    r_hip_pitch=random.random() * 2 * math.pi,
                    l_hip_pitch=random.random() * 2 * math.pi,
                    r_knee=random.random() * 2 * math.pi,
                    l_knee=random.random() * 2 * math.pi,
                    r_ankle_pitch=random.random() * 2 * math.pi,
                    l_ankle_pitch=random.random() * 2 * math.pi,
                    r_ankle_roll=random.random() * 2 * math.pi,
                    l_ankle_roll=random.random() * 2 * math.pi,
                    head_pan=random.random() * 2 * math.pi,
                    head_tilt=random.random() * 2 * math.pi,
                ),
            )


def insert_joint_commands(db: Session, recording_ids: list[int], n: int) -> None:
    for recording_id in recording_ids:
        for i in range(n):
            db.add(
                JointCommand(
                    stamp=float(i),
                    recording_id=recording_id,
                    r_shoulder_pitch=random.random() * 2 * math.pi,
                    l_shoulder_pitch=random.random() * 2 * math.pi,
                    r_shoulder_roll=random.random() * 2 * math.pi,
                    l_shoulder_roll=random.random() * 2 * math.pi,
                    r_elbow=random.random() * 2 * math.pi,
                    l_elbow=random.random() * 2 * math.pi,
                    r_hip_yaw=random.random() * 2 * math.pi,
                    l_hip_yaw=random.random() * 2 * math.pi,
                    r_hip_roll=random.random() * 2 * math.pi,
                    l_hip_roll=random.random() * 2 * math.pi,
                    r_hip_pitch=random.random() * 2 * math.pi,
                    l_hip_pitch=random.random() * 2 * math.pi,
                    r_knee=random.random() * 2 * math.pi,
                    l_knee=random.random() * 2 * math.pi,
                    r_ankle_pitch=random.random() * 2 * math.pi,
                    l_ankle_pitch=random.random() * 2 * math.pi,
                    r_ankle_roll=random.random() * 2 * math.pi,
                    l_ankle_roll=random.random() * 2 * math.pi,
                    head_pan=random.random() * 2 * math.pi,
                    head_tilt=random.random() * 2 * math.pi,
                ),
            )


def insert_game_states(db: Session, recording_ids: list[int], n: int) -> None:
    for recording_id in recording_ids:
        for i in range(n):
            db.add(
                GameState(
                    stamp=float(i),
                    recording_id=recording_id,
                    state=random.choice(list(RobotState)),
                ),
            )


def insert_dummy_data(db: Session, n: int = 10) -> None:
    logger.info("Inserting dummy data...")
    recording_ids: list[int] = insert_recordings(db, n)
    insert_images(db, recording_ids, n)
    insert_rotations(db, recording_ids, n)
    insert_joint_states(db, recording_ids, n)
    insert_joint_commands(db, recording_ids, n)
    insert_game_states(db, recording_ids, n)
    db.commit()
    logger.info(f"Dummy data inserted. Recording IDs: {recording_ids}")
