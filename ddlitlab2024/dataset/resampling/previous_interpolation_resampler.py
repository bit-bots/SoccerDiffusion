from ddlitlab2024.dataset.imports.data import InputData
from ddlitlab2024.dataset.resampling.resampler import Resampler, Sample


class PreviousInterpolationResampler(Resampler):
    def __init__(self, sample_rate_hz: int):
        self.sample_rate_hz = sample_rate_hz
        self.sampling_step_in_seconds = 1 / sample_rate_hz

        self.last_received_data = None
        self.last_sampled_data = None
        self.last_sample_step_timestamp = None

    def resample(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        if self.last_sample_step_timestamp is None:
            return [self._initial_sample(data, relative_timestamp)]
        else:
            return self._samples_until(data, relative_timestamp)

    def _initial_sample(self, data: InputData, relative_timestamp: float) -> Sample[InputData]:
        self.last_received_data = data
        self.last_sampled_data = data
        self.last_sample_step_timestamp = relative_timestamp

        return Sample(data=self.last_sampled_data, timestamp=self.last_sample_step_timestamp)

    def _samples_until(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        assert (
            self.last_received_data is not None
            and self.last_sampled_data is not None
            and self.last_sample_step_timestamp is not None
        ), "There must have been an initial sample"

        samples = []
        num_samples = self._num_passed_sampling_steps(relative_timestamp)

        for _ in range(num_samples):
            if self.is_timestamp_before_next_sampling_step(relative_timestamp):
                self.last_received_data = data

            self.last_sampled_data = self.last_received_data
            self.last_sample_step_timestamp = self.last_sample_step_timestamp + self.sampling_step_in_seconds
            samples.append(Sample(data=self.last_sampled_data, timestamp=self.last_sample_step_timestamp))

        self.last_received_data = data
        return samples

    def _num_passed_sampling_steps(self, relative_timestamp: float) -> int:
        if self.last_sample_step_timestamp is None:
            # There was no previous sample, so it is time to sample once
            return 1

        return int((relative_timestamp - self.last_sample_step_timestamp) / self.sampling_step_in_seconds)

    def is_timestamp_before_next_sampling_step(self, relative_timestamp: float) -> bool:
        if self.last_sample_step_timestamp is None:
            # There was no previous sample, so it is time to sample
            return True

        return relative_timestamp - self.last_sample_step_timestamp <= self.sampling_step_in_seconds
