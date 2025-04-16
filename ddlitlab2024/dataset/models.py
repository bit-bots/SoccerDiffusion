from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, MetaData, String, asc
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import LargeBinary

DEFAULT_IMG_SIZE = (480, 480)


class RobotState(str, Enum):
    PLAYING = "PLAYING"
    POSITIONING = "POSITIONING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def values(cls):
        return sorted([e.value for e in cls])

    def __int__(self):
        # Use index of sorted strings
        return self.values().index(self.value)


class TeamColor(str, Enum):
    BLUE = "BLUE"
    RED = "RED"
    YELLOW = "YELLOW"
    BLACK = "BLACK"
    WHITE = "WHITE"
    GREEN = "GREEN"
    ORANGE = "ORANGE"
    PURPLE = "PURPLE"
    BROWN = "BROWN"
    GRAY = "GRAY"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class Base(DeclarativeBase):
    # Setup consistent naming patterns for constraints, based on suggestions:
    # https://alembic.sqlalchemy.org/en/latest/naming.html
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Recording(Base):
    __tablename__ = "Recording"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    allow_public: Mapped[bool] = mapped_column(Boolean, default=False)
    original_file: Mapped[str] = mapped_column(String, nullable=False)
    team_name: Mapped[str] = mapped_column(String, nullable=False)
    team_color: Mapped[Optional[TeamColor]] = mapped_column(String, nullable=True)
    robot_type: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    img_width: Mapped[int] = mapped_column(Integer, default=DEFAULT_IMG_SIZE[0])
    img_height: Mapped[int] = mapped_column(Integer, default=DEFAULT_IMG_SIZE[1])
    # Scaling factors for original image size to img_width x img_height
    img_width_scaling: Mapped[float] = mapped_column(Float, nullable=False)
    img_height_scaling: Mapped[float] = mapped_column(Float, nullable=False)

    images: Mapped[list["Image"]] = relationship("Image", back_populates="recording", cascade="all, delete-orphan")
    rotations: Mapped[list["Rotation"]] = relationship(
        "Rotation", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_states: Mapped[list["JointStates"]] = relationship(
        "JointStates", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_commands: Mapped[list["JointCommands"]] = relationship(
        "JointCommands", back_populates="recording", cascade="all, delete-orphan"
    )
    game_states: Mapped[list["GameState"]] = relationship(
        "GameState", back_populates="recording", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(img_width > 0, name="img_width_value"),
        CheckConstraint(img_height > 0, name="img_height_value"),
        CheckConstraint(team_color.in_(TeamColor.values()), name="team_color_enum"),
        CheckConstraint(end_time >= start_time, name="end_time_ge_start_time"),
    )

    def duration(self) -> Optional[timedelta]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


class Image(Base):
    __tablename__ = "Image"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    # The image data should contain the image as bytes using an rgb8 format (3 channels) and uint8 type.
    # and should be of size (img_width, img_height) as specified in the recording (default 480x480)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="images")

    __table_args__ = (
        CheckConstraint("stamp >= 0", name="stamp_value"),
        # Index to retrieve images in order from a given recording
        Index(None, "recording_id", asc("stamp")),
    )

    def __init__(
        self, stamp: float, image: np.ndarray, recording_id: int | None = None, recording: Recording | None = None
    ):
        assert image.dtype == np.uint8, "Image must be of type np.uint8"
        assert image.ndim == 3, "Image must have 3 dimensions"
        assert image.shape[2] == 3, "Image must have 3 channels"
        assert recording_id is not None or recording is not None, "Either recording_id or recording must be provided"

        if recording is None:
            super().__init__(stamp=stamp, recording_id=recording_id, data=image.tobytes())
        else:
            super().__init__(stamp=stamp, recording=recording, data=image.tobytes())


class Rotation(Base):
    __tablename__ = "Rotation"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)
    w: Mapped[float] = mapped_column(Float, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="rotations")

    __table_args__ = (
        CheckConstraint(stamp >= 0, name="stamp_value"),
        CheckConstraint("x >= -1 AND x <= 1", name="x_value"),
        CheckConstraint("y >= -1 AND y <= 1", name="y_value"),
        CheckConstraint("z >= -1 AND z <= 1", name="z_value"),
        CheckConstraint("w >= -1 AND w <= 1", name="w_value"),
        # Index to retrieve rotations in order from a given recording
        Index(None, "recording_id", asc("stamp")),
    )


class JointStates(Base):
    __tablename__ = "JointStates"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    r_shoulder_pitch: Mapped[float] = mapped_column(Float, name="RShoulderPitch")
    l_shoulder_pitch: Mapped[float] = mapped_column(Float, name="LShoulderPitch")
    r_shoulder_roll: Mapped[float] = mapped_column(Float, name="RShoulderRoll")
    l_shoulder_roll: Mapped[float] = mapped_column(Float, name="LShoulderRoll")
    # The Wolfgang-OP only has a single elbow joint, which is why they are just called elbow
    # while the Nao has two elbow joints (r_elbow thus represents RElbowRoll)
    r_elbow: Mapped[float] = mapped_column(Float, name="RElbow")
    r_elbow_yaw: Mapped[float] = mapped_column(Float, name="RElbowYaw", default=0.0)
    l_elbow: Mapped[float] = mapped_column(Float, name="LElbow")
    l_elbow_yaw: Mapped[float] = mapped_column(Float, name="LElbowYaw", default=0.0)
    r_hip_yaw: Mapped[float] = mapped_column(Float, name="RHipYaw")
    l_hip_yaw: Mapped[float] = mapped_column(Float, name="LHipYaw")
    r_hip_roll: Mapped[float] = mapped_column(Float, name="RHipRoll")
    l_hip_roll: Mapped[float] = mapped_column(Float, name="LHipRoll")
    r_hip_pitch: Mapped[float] = mapped_column(Float, name="RHipPitch")
    l_hip_pitch: Mapped[float] = mapped_column(Float, name="LHipPitch")
    r_knee: Mapped[float] = mapped_column(Float, name="RKnee")
    l_knee: Mapped[float] = mapped_column(Float, name="LKnee")
    r_ankle_pitch: Mapped[float] = mapped_column(Float, name="RAnklePitch")
    l_ankle_pitch: Mapped[float] = mapped_column(Float, name="LAnklePitch")
    r_ankle_roll: Mapped[float] = mapped_column(Float, name="RAnkleRoll")
    l_ankle_roll: Mapped[float] = mapped_column(Float, name="LAnkleRoll")
    head_pan: Mapped[float] = mapped_column(Float, name="HeadPan")
    head_tilt: Mapped[float] = mapped_column(Float, name="HeadTilt")

    recording: Mapped["Recording"] = relationship("Recording", back_populates="joint_states")

    __table_args__ = (
        CheckConstraint(stamp >= 0, name="stamp_value"),
        CheckConstraint("RShoulderPitch >= 0 AND RShoulderPitch < 2 * pi()", name="RShoulderPitch_value"),
        CheckConstraint("LShoulderPitch >= 0 AND LShoulderPitch < 2 * pi()", name="LShoulderPitch_value"),
        CheckConstraint("RShoulderRoll >= 0 AND RShoulderRoll < 2 * pi()", name="RShoulderRoll_value"),
        CheckConstraint("LShoulderRoll >= 0 AND LShoulderRoll < 2 * pi()", name="LShoulderRoll_value"),
        CheckConstraint("RElbow >= 0 AND RElbow < 2 * pi()", name="RElbow_value"),
        CheckConstraint("RElbowYaw >= 0 AND RElbowYaw < 2 * pi()", name="RElbowYaw_value"),
        CheckConstraint("LElbow >= 0 AND LElbow < 2 * pi()", name="LElbow_value"),
        CheckConstraint("LElbowYaw >= 0 AND LElbowYaw < 2 * pi()", name="LElbowYaw_value"),
        CheckConstraint("RHipYaw >= 0 AND RHipYaw < 2 * pi()", name="RHipYaw_value"),
        CheckConstraint("LHipYaw >= 0 AND LHipYaw < 2 * pi()", name="LHipYaw_value"),
        CheckConstraint("RHipRoll >= 0 AND RHipRoll < 2 * pi()", name="RHipRoll_value"),
        CheckConstraint("LHipRoll >= 0 AND LHipRoll < 2 * pi()", name="LHipRoll_value"),
        CheckConstraint("RHipPitch >= 0 AND RHipPitch < 2 * pi()", name="RHipPitch_value"),
        CheckConstraint("LHipPitch >= 0 AND LHipPitch < 2 * pi()", name="LHipPitch_value"),
        CheckConstraint("RKnee >= 0 AND RKnee < 2 * pi()", name="RKnee_value"),
        CheckConstraint("LKnee >= 0 AND LKnee < 2 * pi()", name="LKnee_value"),
        CheckConstraint("RAnklePitch >= 0 AND RAnklePitch < 2 * pi()", name="RAnklePitch_value"),
        CheckConstraint("LAnklePitch >= 0 AND LAnklePitch < 2 * pi()", name="LAnklePitch_value"),
        CheckConstraint("RAnkleRoll >= 0 AND RAnkleRoll < 2 * pi()", name="RAnkleRoll_value"),
        CheckConstraint("LAnkleRoll >= 0 AND LAnkleRoll < 2 * pi()", name="LAnkleRoll_value"),
        CheckConstraint("HeadPan >= 0 AND HeadPan < 2 * pi()", name="HeadPan_value"),
        CheckConstraint("HeadTilt >= 0 AND HeadTilt < 2 * pi()", name="HeadTilt_value"),
        # Index to retrieve joint states in order from a given recording
        Index(None, "recording_id", asc("stamp")),
    )

    @staticmethod
    def get_ordered_joint_names() -> list[str]:
        return [
            JointStates.head_pan.name,
            JointStates.head_tilt.name,
            JointStates.l_ankle_pitch.name,
            JointStates.l_ankle_roll.name,
            JointStates.l_elbow.name,
            JointStates.l_elbow_yaw.name,
            JointStates.l_hip_pitch.name,
            JointStates.l_hip_roll.name,
            JointStates.l_hip_yaw.name,
            JointStates.l_knee.name,
            JointStates.l_shoulder_pitch.name,
            JointStates.l_shoulder_roll.name,
            JointStates.r_ankle_pitch.name,
            JointStates.r_ankle_roll.name,
            JointStates.r_elbow.name,
            JointStates.r_elbow_yaw.name,
            JointStates.r_hip_pitch.name,
            JointStates.r_hip_roll.name,
            JointStates.r_hip_yaw.name,
            JointStates.r_knee.name,
            JointStates.r_shoulder_pitch.name,
            JointStates.r_shoulder_roll.name,
        ]


class JointCommands(Base):
    __tablename__ = "JointCommands"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    r_shoulder_pitch: Mapped[float] = mapped_column(Float, name="RShoulderPitch")
    l_shoulder_pitch: Mapped[float] = mapped_column(Float, name="LShoulderPitch")
    r_shoulder_roll: Mapped[float] = mapped_column(Float, name="RShoulderRoll")
    l_shoulder_roll: Mapped[float] = mapped_column(Float, name="LShoulderRoll")
    # The Wolfgang-OP only has a single elbow joint, which is why they are just called elbow
    # while the Nao has two elbow joints (r_elbow thus represents RElbowRoll)
    r_elbow: Mapped[float] = mapped_column(Float, name="RElbow")
    r_elbow_yaw: Mapped[float] = mapped_column(Float, name="RElbowYaw", default=0.0)
    l_elbow: Mapped[float] = mapped_column(Float, name="LElbow")
    l_elbow_yaw: Mapped[float] = mapped_column(Float, name="LElbowYaw", default=0.0)
    r_hip_yaw: Mapped[float] = mapped_column(Float, name="RHipYaw")
    l_hip_yaw: Mapped[float] = mapped_column(Float, name="LHipYaw")
    r_hip_roll: Mapped[float] = mapped_column(Float, name="RHipRoll")
    l_hip_roll: Mapped[float] = mapped_column(Float, name="LHipRoll")
    r_hip_pitch: Mapped[float] = mapped_column(Float, name="RHipPitch")
    l_hip_pitch: Mapped[float] = mapped_column(Float, name="LHipPitch")
    r_knee: Mapped[float] = mapped_column(Float, name="RKnee")
    l_knee: Mapped[float] = mapped_column(Float, name="LKnee")
    r_ankle_pitch: Mapped[float] = mapped_column(Float, name="RAnklePitch")
    l_ankle_pitch: Mapped[float] = mapped_column(Float, name="LAnklePitch")
    r_ankle_roll: Mapped[float] = mapped_column(Float, name="RAnkleRoll")
    l_ankle_roll: Mapped[float] = mapped_column(Float, name="LAnkleRoll")
    head_pan: Mapped[float] = mapped_column(Float, name="HeadPan")
    head_tilt: Mapped[float] = mapped_column(Float, name="HeadTilt")

    recording: Mapped["Recording"] = relationship("Recording", back_populates="joint_commands")

    __table_args__ = (
        CheckConstraint(stamp >= 0, name="stamp_value"),
        CheckConstraint("RShoulderPitch >= 0 AND RShoulderPitch < 2 * pi()", name="RShoulderPitch_value"),
        CheckConstraint("LShoulderPitch >= 0 AND LShoulderPitch < 2 * pi()", name="LShoulderPitch_value"),
        CheckConstraint("RShoulderRoll >= 0 AND RShoulderRoll < 2 * pi()", name="RShoulderRoll_value"),
        CheckConstraint("LShoulderRoll >= 0 AND LShoulderRoll < 2 * pi()", name="LShoulderRoll_value"),
        CheckConstraint("RElbow >= 0 AND RElbow < 2 * pi()", name="RElbow_value"),
        CheckConstraint("RElbowYaw >= 0 AND RElbowYaw < 2 * pi()", name="RElbowYaw_value"),
        CheckConstraint("LElbow >= 0 AND LElbow < 2 * pi()", name="LElbow_value"),
        CheckConstraint("LElbowYaw >= 0 AND LElbowYaw < 2 * pi()", name="LElbowYaw_value"),
        CheckConstraint("RHipYaw >= 0 AND RHipYaw < 2 * pi()", name="RHipYaw_value"),
        CheckConstraint("LHipYaw >= 0 AND LHipYaw < 2 * pi()", name="LHipYaw_value"),
        CheckConstraint("RHipRoll >= 0 AND RHipRoll < 2 * pi()", name="RHipRoll_value"),
        CheckConstraint("LHipRoll >= 0 AND LHipRoll < 2 * pi()", name="LHipRoll_value"),
        CheckConstraint("RHipPitch >= 0 AND RHipPitch < 2 * pi()", name="RHipPitch_value"),
        CheckConstraint("LHipPitch >= 0 AND LHipPitch < 2 * pi()", name="LHipPitch_value"),
        CheckConstraint("RKnee >= 0 AND RKnee < 2 * pi()", name="RKnee_value"),
        CheckConstraint("LKnee >= 0 AND LKnee < 2 * pi()", name="LKnee_value"),
        CheckConstraint("RAnklePitch >= 0 AND RAnklePitch < 2 * pi()", name="RAnklePitch_value"),
        CheckConstraint("LAnklePitch >= 0 AND LAnklePitch < 2 * pi()", name="LAnklePitch_value"),
        CheckConstraint("RAnkleRoll >= 0 AND RAnkleRoll < 2 * pi()", name="RAnkleRoll_value"),
        CheckConstraint("LAnkleRoll >= 0 AND LAnkleRoll < 2 * pi()", name="LAnkleRoll_value"),
        CheckConstraint("HeadPan >= 0 AND HeadPan < 2 * pi()", name="HeadPan_value"),
        CheckConstraint("HeadTilt >= 0 AND HeadTilt < 2 * pi()", name="HeadTilt_value"),
        # Index to retrieve joint commands in order from a given recording
        Index(None, "recording_id", asc("stamp")),
    )


class GameState(Base):
    __tablename__ = "GameState"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    state: Mapped[RobotState] = mapped_column(String, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="game_states")

    __table_args__ = (
        CheckConstraint(state.in_(RobotState.values()), name="state_enum"),
        # Index to retrieve game states in order from a given recording
        Index(None, "recording_id", asc("stamp")),
    )


def stamp_to_seconds_nanoseconds(stamp: float) -> tuple[int, int]:
    seconds = int(stamp // 1)
    nanoseconds = int((stamp % 1) * 1e9)
    return seconds, nanoseconds


def stamp_to_nanoseconds(stamp: float) -> int:
    return int(stamp * 1e9)
