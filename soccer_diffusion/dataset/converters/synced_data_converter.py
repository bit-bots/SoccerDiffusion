from soccer_diffusion.dataset.converters.converter import Converter
from soccer_diffusion.dataset.imports.data import InputData, ModelData
from soccer_diffusion.dataset.models import JointCommands, JointStates, Recording, Rotation
from soccer_diffusion.dataset.resampling.previous_interpolation_resampler import PreviousInterpolationResampler
from soccer_diffusion.utils.utils import shift_radian_to_positive_range


class SyncedDataConverter(Converter):
    def __init__(self, resampler: PreviousInterpolationResampler) -> None:
        self.resampler = resampler

    def populate_recording_metadata(self, data: InputData, recording: Recording):
        pass

    def convert_to_model(self, data: InputData, relative_timestamp: float, recording: Recording) -> ModelData:
        assert data.joint_state is not None, "joint_states are required in synced resampling data"
        assert all(
            command is not None for command in data.joint_command.values()
        ), "joint_commands are required in synced resampling data"
        assert data.rotation is not None, "IMU rotation is required in synced resampling data"

        models = ModelData()

        for sample in self.resampler.resample(data, relative_timestamp):
            models.rotations.append(self._create_rotation(sample.data.rotation, sample.timestamp, recording))
            models.joint_states.append(self._create_joint_states(sample.data.joint_state, sample.timestamp, recording))
            models.joint_commands.append(
                self._create_joint_commands(sample.data.joint_command, sample.timestamp, recording)
            )

        return models

    def _create_rotation(self, msg, sampling_timestamp: float, recording: Recording) -> Rotation:
        return Rotation(
            stamp=sampling_timestamp,
            recording=recording,
            x=msg.x,
            y=msg.y,
            z=msg.z,
            w=msg.w,
        )

    def _create_joint_states(self, joint_states_data, sampling_timestamp: float, recording: Recording) -> JointStates:
        shifted_joint_states = {}

        for joint, position in joint_states_data.items():
            shifted_joint_states[joint] = shift_radian_to_positive_range(position)

        return JointStates(stamp=sampling_timestamp, recording=recording, **shifted_joint_states)

    def _create_joint_commands(
        self, joint_commands_data, sampling_timestamp: float, recording: Recording
    ) -> JointCommands:
        shifted_joint_commands = {}

        for joint, command in joint_commands_data.items():
            shifted_joint_commands[joint] = shift_radian_to_positive_range(command)

        return JointCommands(stamp=sampling_timestamp, recording=recording, **shifted_joint_commands)
