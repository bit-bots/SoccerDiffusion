#!/usr/bin/env python
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import sqlite3
import pandas as pd
from ddlitlab2024.dataset.models import JointCommand, JointState, Recording


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

    def __getitem__(self, idx):
        # Find the recording that contains the sample
        for start_sample, end_sample, recording_id in self.sample_boundaries:
            if idx >= start_sample and idx < end_sample:
                boundary = (recording_id, start_sample)
                break
        assert boundary is not None, "Could not find the recording that contains the sample"
        recording_id, start_sample = boundary

        # Calculate the sample index in the recording
        sample_index = idx - start_sample

        # Get the joint command
        raw_joint_command = pd.read_sql_query(
            f"SELECT * FROM JointCommand WHERE recording_id = {recording_id} ORDER BY stamp ASC LIMIT {self.num_samples_joint_trajectory_future} OFFSET {sample_index}",
            self.db_connection,
        )

        # TODO make index CREATE INDEX idx_recording_stamp_joint_command ON JointCommand(recording_id, stamp)

        # Convert to numpy array, keep only the joint angle columns (np.float32 type)
        raw_joint_command = raw_joint_command.drop(columns=["stamp", "recording_id"]).to_numpy(dtype=np.float32)

        return torch.from_numpy(raw_joint_command)


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

    import time

    time1 = time.time()
    for i, _ in enumerate(dataset):
        if i == 10:
            break

    print((time.time() - time1) / 10)

    # print(len(sample))
    # plt.plot(sample["stamp"], sample["LKnee"])
    # plt.show()
