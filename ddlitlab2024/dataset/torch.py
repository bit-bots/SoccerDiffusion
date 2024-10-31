# Torch dataset class for the DDLIT Lab 2024 dataset

import torch
import sqlite3
import pandas as pd
from torch.utils.data import Dataset
from ddlitlab2024.dataset.models import JointCommand, JointState, IMU, Recording, Game, Video


class DDLITLab2024Dataset(Dataset):
    def __init__(self,
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
        self.data_base_path = data_base_path

        # Load the data from the database
        self.db_connection = sqlite3.connect(self.data_base_path)

        # Lock the database to prevent writing
        self.db_connection.execute("PRAGMA locking_mode = EXCLUSIVE")

        # Calculate the length of a batch
        self.sample_length_s = self.num_samples_joint_trajectory_future / self.sample_rate_joint_trajectory

        # Stride in seconds
        self.stride_s = self.trajectory_stride / self.sample_rate_joint_trajectory

        # Get the total length of the dataset in seconds
        cursor = self.db_connection.cursor()

        # SQL query that get the first and last timestamp of the joint command for each recording
        cursor.execute("SELECT recording_id, MIN(stamp), MAX(stamp) FROM joint_command GROUP BY recording_id")
        recording_timestamps = cursor.fetchall()

        # Calculate how many batches can be build from each recording
        self.num_samples = 0
        self.sample_boundaries = []
        for recording_id, start_timestamp, end_timestamp in recording_timestamps:
            recording_length = end_timestamp - start_timestamp
            assert recording_length > 0, "Recording length is negative"
            total_samples_before = self.num_samples
            # Calculate the number of batches that can be build from the recording including the stride
            self.num_samples += int((recording_length - self.sample_length_s) / self.stride_s)
            # Store the boundaries of the samples for later retrieval
            self.sample_boundaries.append((total_samples_before, self.num_samples, recording_id, start_timestamp))
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Find the recording that contains the sample
        boundary = None 
        for start_sample, end_sample, recording_id, st in self.sample_boundaries:
            if idx >= start_sample and idx < end_sample:
                boundary = (recording_id, st)
                break
        assert boundary is not None, "Could not find the recording that contains the sample"
        recording_id, start_timestamp = boundary
        
        # Calculate the timestamp of the sample
        sample_timestamp = start_timestamp + (idx - boundary[0]) * self.stride_s
        sample_timestamp_future = sample_timestamp + self.num_samples_joint_trajectory_future / self.sample_rate_joint_trajectory

        # Get the joint command
        raw_joint_command = pd.read_sql_query(f"SELECT * FROM joint_command WHERE recording_id = {recording_id} AND stamp >= {sample_timestamp} AND stamp < {sample_timestamp_future}", self.db_connection)

        



