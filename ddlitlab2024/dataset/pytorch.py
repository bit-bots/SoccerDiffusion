#!/usr/bin/env python
import os
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from ddlitlab2024 import DB_PATH
from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.models import JointStates, RobotState
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder
from ddlitlab2024.utils.utils import quats_to_5d


def connect_to_db(data_base_path: str | Path = DB_PATH, worker_id: int | None = None) -> sqlite3.Connection:
    logger.info(f"Connecting to database at {data_base_path} in worker {worker_id}")
    # The Data exists in a sqlite database
    data_base_path = str(data_base_path)
    assert data_base_path.endswith(".sqlite3"), "The database should be a sqlite file"
    assert os.path.exists(data_base_path), f"The database file '{data_base_path}' does not exist"

    return sqlite3.connect(f"file:{data_base_path}?immutable=1", uri=True)  # Open the database in read-only mode


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # The dataset copy in the worker process
    dataset.db_connection = connect_to_db(worker_id=worker_id)


class DDLITLab2024Dataset(Dataset):
    @dataclass
    class Result:
        joint_command: torch.Tensor
        joint_command_history: Optional[torch.Tensor]
        joint_state: Optional[torch.Tensor]
        rotation: Optional[torch.Tensor]
        game_state: Optional[torch.Tensor]
        robot_type: Optional[torch.Tensor]
        image_data: Optional[torch.Tensor]
        image_stamps: Optional[torch.Tensor]

        def shapes(self) -> dict[str, tuple[int, ...]]:
            return {k: v.shape for k, v in asdict(self).items() if v is not None}

    def __init__(
        self,
        db_connection: sqlite3.Connection | None = None,
        num_samples_imu: int = 100,
        imu_representation: IMUEncoder.OrientationEmbeddingMethod = IMUEncoder.OrientationEmbeddingMethod.QUATERNION,
        num_samples_joint_states: int = 100,
        num_samples_joint_trajectory: int = 100,
        num_samples_joint_trajectory_future: int = 10,
        sampling_rate: int = 100,
        max_fps_video: int = 10,
        num_frames_video: int = 50,
        image_resolution: int = 480,
        trajectory_stride: int = 10,
        num_joints: int = 20,
        use_images: bool = True,
        use_image_embeddings: bool = True,
        use_imu: bool = True,
        use_joint_states: bool = True,
        use_action_history: bool = True,
        use_game_state: bool = True,
        use_robot_type: bool = False,
        use_cache: bool = True,
    ):
        # Initialize the database connection
        self.db_connection: sqlite3.Connection = db_connection if db_connection else connect_to_db()

        # Store the parameters
        self.num_samples_imu = num_samples_imu
        self.imu_representation = imu_representation
        self.num_samples_joint_states = num_samples_joint_states
        self.num_samples_joint_trajectory = num_samples_joint_trajectory
        self.num_samples_joint_trajectory_future = num_samples_joint_trajectory_future
        self.sampling_rate = sampling_rate
        self.max_fps_video = max_fps_video
        self.num_frames_video = num_frames_video
        self.image_resolution = image_resolution
        self.trajectory_stride = trajectory_stride
        self.num_joints = num_joints
        self.joint_names = JointStates.get_ordered_joint_names()
        self.use_images = use_images
        self.use_image_embeddings = use_image_embeddings
        self.use_imu = use_imu
        self.use_joint_states = use_joint_states
        self.use_action_history = use_action_history
        self.use_game_state = use_game_state
        self.use_robot_type = use_robot_type

        # Memory cache for the dataset
        self.use_cache = use_cache
        self.cache = {}

        # Print out metadata
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT team_name, start_time, location, original_file FROM Recording")
        recordings = cursor.fetchall()
        table = tabulate(recordings, headers=["Team name", "Start time", "Location", "Original file"])
        logger.info(f"Using the following recordings:\n{table}")

        # SQL query that get the first and last timestamp of the joint command for each recording
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT recording_id, COUNT(*) AS num_entries_in_recording FROM JointCommands GROUP BY recording_id"
        )
        recording_timestamps = cursor.fetchall()

        # Calculate how many batches can be build from each recording
        self.num_samples = 0
        self.sample_boundaries = []
        for recording_id, num_data_points in recording_timestamps:
            assert num_data_points > 0, "Recording length is negative or zero"
            total_samples_before = self.num_samples
            # Calculate the number of batches that can be build from the recording including the stride
            self.num_samples += int(
                (num_data_points - self.num_samples_joint_trajectory_future) / self.trajectory_stride
            )
            # Store the boundaries of the samples for later retrieval
            self.sample_boundaries.append((total_samples_before, self.num_samples, recording_id))

    def __len__(self):
        return self.num_samples

    def query_joint_data(
        self, recording_id: int, start_sample: int, num_samples: int, table: Literal["JointCommands", "JointStates"]
    ) -> torch.Tensor:
        # Get the joint state
        raw_joint_data = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE recording_id = {recording_id} "
            f"ORDER BY stamp ASC LIMIT {num_samples} OFFSET {start_sample}",
            # TODO other direction  TODO make params correct
            self.db_connection,
        )

        # Convert to numpy array, keep only the joint angle columns in alphabetical order
        raw_joint_data = raw_joint_data[self.joint_names].to_numpy(dtype=np.float32)

        assert raw_joint_data.shape[1] == self.num_joints, "The number of joints is not correct"

        # We don't need padding here, because we sample the data in the correct length for the targets
        return torch.from_numpy(raw_joint_data)

    def query_joint_data_history(
        self, recording_id: int, end_sample: int, num_samples: int, table: Literal["JointCommands", "JointStates"]
    ) -> torch.Tensor:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples_to_query = end_sample - start_sample

        # Get the joint data
        raw_joint_data = self.query_joint_data(recording_id, start_sample, num_samples_to_query, table)

        # Pad the data if necessary, for the input data / history it might be necessary
        # during the startup / first samples
        # Zero pad the joint state if the number of samples is less than the required number of samples
        if raw_joint_data.shape[0] < num_samples:
            raw_joint_data = torch.cat(
                (
                    torch.zeros(
                        (num_samples - raw_joint_data.shape[0], raw_joint_data.shape[1]), dtype=raw_joint_data.dtype
                    ),
                    raw_joint_data,
                ),
                dim=0,
            )
            assert raw_joint_data.shape[0] == num_samples, "The padded array is not the correct shape"
            assert raw_joint_data[0, 0] == 0.0, "The array is not zero padded"

        return raw_joint_data

    def query_image_data(
        self, recording_id: int, end_time_stamp: float, context_len: float, num_frames: int, resolution: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get the image data
        cursor = self.db_connection.cursor()

        col = "embedding" if self.use_image_embeddings else "data"
        cursor.execute(
            # Select the last num_samples images before the current time stamp
            # and order them by time stamp in ascending order
            f"SELECT stamp, {col} FROM Image "
            "WHERE recording_id = $1 AND stamp BETWEEN $2 - $3 AND $2 ORDER BY stamp ASC;",
            (recording_id, end_time_stamp, context_len),
        )

        # Get the image data
        response = cursor.fetchall()

        # Drop the first frames if there are more than num_frames
        if len(response) > num_frames:
            response = response[-num_frames:]

        # Get the image data
        stamps = []
        image_data = []

        # Define the preprocessing pipeline
        preprocessing = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        if self.use_image_embeddings:
            # Decode image embeddings
            for stamp, data in response:
                data = torch.tensor(np.frombuffer(data, dtype=np.float32).reshape(-1))
                assert data.shape[0] == 384, "Embedding size might be wrong"
                image_data.append(data)
                stamps.append(stamp)
        else:
            # Decode the raw image data
            for stamp, data in response:
                # Deserialize the image data
                image = np.frombuffer(data, dtype=np.uint8).reshape(480, 480, 3)
                # Resize the image
                image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
                # Apply the preprocessing pipeline
                image = preprocessing(image)
                # Append to the list
                image_data.append(image)
                stamps.append(stamp)

        # Apply zero padding if necessary
        if len(image_data) < num_frames:
            if self.use_image_embeddings:
                single_pad = torch.zeros(384, dtype=torch.float32)
            else:
                single_pad = torch.zeros((3, resolution, resolution), dtype=torch.float32)

            image_data = [
                single_pad
                for _ in range(num_frames - len(image_data))
            ] + image_data
            stamps = [end_time_stamp - context_len for _ in range(num_frames - len(stamps))] + stamps

        image_data = torch.stack(image_data, axis=0)
        stamps = torch.tensor(stamps)

        return stamps, image_data

    def query_imu_data(self, recording_id: int, end_sample: int, num_samples: int) -> torch.Tensor:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples_to_query = end_sample - start_sample

        # Get the imu data
        raw_imu_data = pd.read_sql_query(
            f"SELECT * FROM Rotation WHERE recording_id = {recording_id} "
            f"ORDER BY stamp ASC LIMIT {num_samples_to_query} OFFSET {start_sample}",
            # TODO make params correct
            self.db_connection,
        )

        # Convert to numpy array
        raw_imu_data = raw_imu_data[["x", "y", "z", "w"]].to_numpy(dtype=np.float32)

        # Add padding if necessary (identity quaternion)
        if raw_imu_data.shape[0] < num_samples:
            identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            raw_imu_data = np.concatenate(
                (
                    np.tile(identity_quaternion, (num_samples - raw_imu_data.shape[0], 1)),
                    raw_imu_data,
                ),
                axis=0,
            )

            assert raw_imu_data.shape[0] == num_samples, "The padded array is not the correct shape"
            assert np.allclose(
                raw_imu_data[0], identity_quaternion
            ), "The array does not start with the identity quaternion, even though it is padded"

        # Convert to correct representation
        match self.imu_representation:
            case IMUEncoder.OrientationEmbeddingMethod.FIVE_DIM:
                imu_data = quats_to_5d(raw_imu_data)

            case IMUEncoder.OrientationEmbeddingMethod.QUATERNION:
                imu_data = raw_imu_data

            case rep:
                raise NotImplementedError(f"Unknown IMU representation {rep}")

        return torch.from_numpy(imu_data).float()

    def query_current_game_state(self, recording_id: int, stamp: float) -> torch.Tensor:
        cursor = self.db_connection.cursor()
        # Select last game state before the current stamp
        cursor.execute(
            "SELECT state FROM GameState WHERE recording_id = $1 AND stamp <= $2 ORDER BY stamp DESC LIMIT 1",
            (recording_id, stamp),
        )

        # Get the game state
        game_state: RobotState = cursor.fetchone()

        # If no game state is found set it to unknown
        if game_state is None:
            game_state = RobotState.UNKNOWN
        else:
            game_state = RobotState(game_state[0])

        return torch.tensor(int(game_state))

    def query_robot_type(self, recording_id: int) -> torch.Tensor:
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT robot_type FROM Recording WHERE _id = $1",
            (recording_id,),
        )
        robot_type = cursor.fetchone()
        robot_types = ["NAO6", "Wolfgang-OP"]
        assert robot_type is not None, "The robot type is not set in the database"
        assert robot_type[0] in robot_types, f"The robot type '{robot_type[0]}' is not supported"
        robot_type_id = robot_types.index(robot_type[0])
        return torch.tensor(robot_type_id)

    def __getitem__(self, idx: int) -> Result:
        # Check if the index is in the cache
        if idx in self.cache:
            return self.cache[idx]

        # Find the recording that contains the sample
        for start_sample, end_sample, recording_id in self.sample_boundaries:
            if idx >= start_sample and idx < end_sample:
                boundary = (recording_id, start_sample)
                break
        assert boundary is not None, "Could not find the recording that contains the sample"
        recording_id, start_sample = boundary

        # We assume that joint command, imu and joint state have the sampling rate and are synchronized
        # Game state and image data are not synchronized with the other data

        # Calculate the sample index in the recording
        sample_index = int(idx - start_sample)
        # Calculate the index of the joint command where this sample starts
        # We assume that the joint command, joint state and imu data are roughly synchronized
        # Therefore we can use the joint command index as a reference
        sample_joint_command_index = sample_index * self.trajectory_stride
        # Calculate the time stamp of the sample
        stamp = sample_joint_command_index / self.sampling_rate

        # Get the image data
        if self.use_images:
            image_stamps, image_data = self.query_image_data(
                recording_id,
                stamp,
                # The duration is used to narrow down the query for a faster retrieval,
                # so we consider it as an upper bound
                (self.num_frames_video + 1) / self.max_fps_video,
                self.num_frames_video,
                self.image_resolution,
            )
            # Some sanity checks
            assert all([stamp >= image_stamp for image_stamp in image_stamps]), "The image data is not synchronized"
            assert len(image_stamps) == self.num_frames_video, "The image data is not the correct length"
            if self.use_image_embeddings:
                assert image_data.shape[0] == self.num_frames_video
            else:
                assert image_data.shape == (
                    self.num_frames_video,
                    3,
                    self.image_resolution,
                    self.image_resolution,
                ), "The image data has the wrong shape"
            assert (
                image_stamps[0] >= stamp - (self.num_frames_video + 1) / self.max_fps_video
            ), "The image data is not synchronized"
        else:
            image_stamps, image_data = None, None

        # Get the joint command target (future)
        joint_command = self.query_joint_data(
            recording_id, sample_joint_command_index, self.num_samples_joint_trajectory_future, "JointCommands"
        )
        assert len(joint_command) == self.num_samples_joint_trajectory_future, "The joint command has the wrong length"

        # Get the joint command history
        if self.use_action_history:
            joint_command_history = self.query_joint_data_history(
                recording_id, sample_joint_command_index, self.num_samples_joint_trajectory, "JointCommands"
            )
        else:
            joint_command_history = None

        # Get the joint state
        if self.use_joint_states:
            joint_state = self.query_joint_data_history(
                recording_id, sample_joint_command_index, self.num_samples_joint_states, "JointStates"
            )
        else:
            joint_state = None

        # Get the robot rotation (IMU data)
        if self.use_imu:
            robot_rotation = self.query_imu_data(recording_id, sample_joint_command_index, self.num_samples_imu)
        else:
            robot_rotation = None

        # Get the game state
        if self.use_game_state:
            game_state = self.query_current_game_state(recording_id, stamp)
        else:
            game_state = None

        if self.use_robot_type:
            robot_type = self.query_robot_type(recording_id)
        else:
            robot_type = None

        # Make a result object
        result = self.Result(
            joint_command=joint_command,
            joint_command_history=joint_command_history,
            joint_state=joint_state,
            image_data=image_data,
            image_stamps=image_stamps,
            rotation=robot_rotation,
            game_state=game_state,
            robot_type=robot_type,
        )

        # Cache the result if necessary
        if self.use_cache:
            self.cache[idx] = result
        return result

    @staticmethod
    def collate_fn(batch: Iterable[Result]) -> Result:
        return DDLITLab2024Dataset.Result(
            joint_command=torch.stack([x.joint_command for x in batch]),
            joint_command_history=torch.stack([x.joint_command_history for x in batch])
            if batch[0].joint_command_history is not None
            else None,
            joint_state=torch.stack([x.joint_state for x in batch]) if batch[0].joint_state is not None else None,
            image_data=torch.stack([x.image_data for x in batch]) if batch[0].image_data is not None else None,
            image_stamps=torch.stack([x.image_stamps for x in batch]) if batch[0].image_stamps is not None else None,
            rotation=torch.stack([x.rotation for x in batch]) if batch[0].rotation is not None else None,
            game_state=torch.tensor([x.game_state for x in batch]) if batch[0].game_state is not None else None,
            robot_type=torch.tensor([x.robot_type for x in batch]) if batch[0].robot_type is not None else None,
        )


class Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    @classmethod
    def fit(cls, data: torch.Tensor):
        return cls(data.mean(dim=0), data.std(dim=0) + 1e-5)

    def normalize(self, data: torch.Tensor):
        return (data - self.mean) / self.std

    def denormalize(self, data: torch.Tensor):
        return data * self.std + self.mean


# Some dummy code to test the dataset
if __name__ == "__main__":
    dataset = DDLITLab2024Dataset()

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, collate_fn=DDLITLab2024Dataset.collate_fn, worker_init_fn=worker_init_fn
    )

    # Plot the first sample
    import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    time1 = time.time()

    batch_0: DDLITLab2024Dataset.Result = next(dataloader.__iter__())

    time2 = time.time()
    print(time2 - time1)

    print(batch_0.shapes())

    for i in range(batch_0.joint_command.shape[0]):
        print(f"\n\n---- Element {i} ----\n")

        # Print the game state
        print(f"Gamestate: {RobotState.values()[batch_0.game_state[i]]}")

        # Plot the image data history as grid 10x5
        fig = plt.figure(figsize=(10, 10))
        plt.title("Image context")
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(5, 10),  # creates 2x2 grid of Axes
            axes_pad=0.1,  # pad between Axes in inch.
        )
        for ax, im in zip(grid, batch_0.image_data[i]):
            ax.imshow(im.permute(1, 2, 0).numpy())
        plt.show()

        # Plot the joint command history and future (a subplot for each joint)
        plt.figure(figsize=(10, 10))
        plt.title("Joint states, commands history and future commands")
        for j in range(20):
            plt.subplot(5, 4, j + 1)
            plt.title(f"Joint {j}")

            # Plot the joint command history
            plt.plot(
                np.arange(-batch_0.joint_command_history.shape[1], 0) / dataset.sampling_rate,
                batch_0.joint_command_history[i, :, j].numpy(),
                label="Command history",
            )

            # Draw the future joint commands
            plt.plot(
                np.arange(batch_0.joint_command.shape[1]) / dataset.sampling_rate,
                batch_0.joint_command[i, :, j].numpy(),
                label="Command future",
            )

            # Draw the joint state history
            plt.plot(
                np.arange(-batch_0.joint_state.shape[1], 0) / dataset.sampling_rate,
                batch_0.joint_state[i, :, j].numpy(),
                label="Joint state",
            )
        plt.legend()
        plt.show()

        # Plot the rotation data
        plt.title("Rotation")
        plt.plot(np.arange(-batch_0.rotation.shape[1], 0) / dataset.sampling_rate, batch_0.rotation[i].numpy())
        plt.legend(["x", "y", "z", "w"])
        plt.show()
