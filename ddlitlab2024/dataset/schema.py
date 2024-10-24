from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship, sessionmaker
from sqlalchemy.types import LargeBinary

from ddlitlab2024.dataset import logger

logger.info("Creating database schema")

Base = declarative_base()


class RobotState(str, Enum):
    POSITIONING = "POSITIONING"
    PLAYING = "PLAYING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def values(cls):
        return [cls.POSITIONING, cls.PLAYING, cls.STOPPED, cls.UNKNOWN]


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
        return [
            cls.BLUE,
            cls.RED,
            cls.YELLOW,
            cls.BLACK,
            cls.WHITE,
            cls.GREEN,
            cls.ORANGE,
            cls.PURPLE,
            cls.BROWN,
            cls.GRAY,
        ]


class Recording(Base):
    __tablename__ = "Recording"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    allow_public: Mapped[bool] = mapped_column(Boolean, default=False)
    original_file: Mapped[str] = mapped_column(String, nullable=False)
    team_name: Mapped[str] = mapped_column(String, nullable=False)
    team_color: Mapped[Optional[TeamColor]] = mapped_column(String, nullable=True)
    robot_type: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    img_width: Mapped[int] = mapped_column(Integer, default=480)
    img_height: Mapped[int] = mapped_column(Integer, default=480)
    img_width_scaling: Mapped[float] = mapped_column(Float, nullable=False)
    img_height_scaling: Mapped[float] = mapped_column(Float, nullable=False)

    images: Mapped[List["Image"]] = relationship("Image", back_populates="recording", cascade="all, delete-orphan")
    rotations: Mapped[List["Rotation"]] = relationship(
        "Rotation", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_states: Mapped[List["JointState"]] = relationship(
        "JointState", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_commands: Mapped[List["JointCommand"]] = relationship(
        "JointCommand", back_populates="recording", cascade="all, delete-orphan"
    )
    game_states: Mapped[List["GameState"]] = relationship(
        "GameState", back_populates="recording", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(img_width > 0),
        CheckConstraint(img_height > 0),
        CheckConstraint(team_color.in_(TeamColor.values())),
    )


class Image(Base):
    __tablename__ = "Image"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="images")

    __table_args__ = (CheckConstraint("stamp >= 0"),)


class Rotation(Base):
    __tablename__ = "Rotation"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    roll: Mapped[float] = mapped_column(Float, nullable=False)
    pitch: Mapped[float] = mapped_column(Float, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="rotations")

    __table_args__ = (
        CheckConstraint("stamp >= 0"),
        CheckConstraint("roll >= 0 AND roll < 2 * pi()"),
        CheckConstraint("pitch >= 0 AND pitch < 2 * pi()"),
    )


class JointState(Base):
    __tablename__ = "JointState"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="joint_states")

    __table_args__ = (CheckConstraint("stamp >= 0"),)


class JointCommand(Base):
    __tablename__ = "JointCommand"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="joint_commands")

    __table_args__ = (CheckConstraint("stamp >= 0"),)


class GameState(Base):
    __tablename__ = "GameState"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    state: Mapped[RobotState] = mapped_column(String, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="game_states")

    __table_args__ = (CheckConstraint(state.in_(RobotState.values())),)


engine = create_engine("sqlite:///data.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

logger.info("Database schema created")
