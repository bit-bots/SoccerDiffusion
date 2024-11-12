#!/usr/bin/env python
import os
import sqlite3
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DDLITLab2024Dataset(Dataset):
    def __init__(
        self,
        data_base_path: str,
        sample_rate_imu: int = 100,
        num_samples_imu: int = 100,
        sample_rate_joint_states: int = 100,
        num_samples_joint_states: int = 100,
        sample_rate_joint_trajectory: int = 100,
        num_samples_joint_trajectory: int = 100,
        num_samples_joint_trajectory_future: int = 10,
        max_fps_video: int = 10,
        num_frames_video: int = 50,
        trajectory_stride: int = 10,
    ):
        # Store the parameters
        self.sample_rate_imu = sample_rate_imu
        self.num_samples_imu = num_samples_imu
        self.sample_rate_joint_states = sample_rate_joint_states
        self.num_samples_joint_states = num_samples_joint_states
        self.sample_rate_joint_trajectory = sample_rate_joint_trajectory
        self.num_samples_joint_trajectory = num_samples_joint_trajectory
        self.num_samples_joint_trajectory_future = num_samples_joint_trajectory_future
        self.max_fps_video = max_fps_video
        self.num_frames_video = num_frames_video
        self.trajectory_stride = trajectory_stride

        # The Data exists in a sqlite database
        assert data_base_path.endswith(".sqlite3"), "The database should be a sqlite file"
        assert os.path.exists(data_base_path), "The database file does not exist"
        self.data_base_path = data_base_path

        # Load the data from the database
        self.db_connection = sqlite3.connect(self.data_base_path)

        # Lock the database to prevent writing
        self.db_connection.execute("PRAGMA locking_mode = EXCLUSIVE")

        # Calculate the length of a batch
        self.sample_length_s = self.num_samples_joint_trajectory_future / self.sample_rate_joint_trajectory

        # Get the total length of the dataset in seconds
        cursor = self.db_connection.cursor()

        # SQL query that get the first and last timestamp of the joint command for each recording
        cursor.execute(
            "SELECT recording_id, COUNT(*) AS num_entries_in_recording FROM JointCommand GROUP BY recording_id"
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
        self, recording_id: int, start_sample: int, num_samples: int, table: Literal["JointCommand", "JointState"]
    ) -> torch.Tensor:
        # Get the joint state
        raw_joint_data = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE recording_id = {recording_id} "
            f"ORDER BY stamp ASC LIMIT {num_samples} OFFSET {start_sample}",
            # TODO other direction  TODO make params correct
            self.db_connection,
        )

        # Convert to numpy array, keep only the joint angle columns (np.float32 type)
        raw_joint_data = raw_joint_data.drop(columns=["_id", "stamp", "recording_id"]).to_numpy(dtype=np.float32)

        # We don't need padding here, because we sample the data in the correct length for the targets
        return torch.from_numpy(raw_joint_data)

    def query_joint_data_history(
        self, recording_id: int, end_sample: int, num_samples: int, table: Literal["JointCommand", "JointState"]
    ) -> torch.Tensor:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples = end_sample - start_sample

        # Get the joint data
        raw_joint_data = self.query_joint_data(recording_id, start_sample, num_samples, table)

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
        self, recording_id: int, end_sample: int, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples = end_sample - start_sample

        # Get the image data
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT stamp, data FROM Image WHERE recording_id = $1 ORDER BY stamp ASC LIMIT $2 OFFSET $3",
            (recording_id, num_samples, start_sample),
        )

        stamps = []
        image_data = []

        # Get the raw image data
        for stamp, data in cursor:
            # Deserialize the image data
            image_data.append(np.frombuffer(data, dtype=np.uint8).reshape(480, 480, 3))
            stamps.append(stamp)

        # Convert to tensor
        image_data = torch.from_numpy(np.stack(image_data, axis=0))
        stamps = torch.tensor(stamps)

        # TODO maybe add padding if the number of samples is less than the required number of samples

        return stamps, image_data

    def __getitem__(self, idx):
        # Find the recording that contains the sample
        for start_sample, end_sample, recording_id in self.sample_boundaries:
            if idx >= start_sample and idx < end_sample:
                boundary = (recording_id, start_sample)
                break
        assert boundary is not None, "Could not find the recording that contains the sample"
        recording_id, start_sample = boundary

        # Calculate the sample index in the recording
        sample_index = int(idx - start_sample)

        # Get the joint command target (future)
        joint_command = self.query_joint_data(
            recording_id, sample_index, self.num_samples_joint_trajectory_future, "JointCommand"
        )

        # Get the joint command history
        joint_command_history = self.query_joint_data_history(
            recording_id, sample_index, self.num_samples_joint_trajectory, "JointCommand"
        )

        # Get the joint state
        joint_state = self.query_joint_data_history(
            recording_id, sample_index, self.num_samples_joint_states, "JointState"
        )

        # Get the robot rotation (IMU data)
        # robot_rotation = self.

        # Get the image data
        image_stamps, image_data = self.query_image_data(recording_id, sample_index, self.num_frames_video)

        # TODO image data, imu data, ...

        return joint_command, joint_command_history, joint_state, image_data, image_stamps


# Some dummy code to test the dataset
if __name__ == "__main__":
    dataset = DDLITLab2024Dataset(os.path.join(os.path.dirname(__file__), "db.sqlite3"))

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x[None, ...] for x in batch]),
        }

    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)

    # Plot the first sample
    import matplotlib.pyplot as plt

    indices = np.random.choice(len(dataset), 100)

    import time

    time1 = time.time()
    for i in indices:
        sample = dataset[i]

    print(time.time() - time1)

    print(sample)

    print(len(sample))
    # Plot joint history and future
    plt.plot(range(100, 110), sample[0].numpy(), label="Future")
    plt.plot(range(100), sample[1].numpy(), label="History")
    plt.legend()
    plt.show()

    plt.imshow(np.hstack(sample[3].numpy()))
    plt.show()
