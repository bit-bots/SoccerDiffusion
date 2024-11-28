from ddlitlab2024.dataset.imports.model_importer import InputData, Sample


class MaxRateResampler:
    def __init__(self, max_sample_rate_hz: int):
        self.max_sample_rate_hz = max_sample_rate_hz
        self.sampling_step_in_seconds = 1 / max_sample_rate_hz

        self.last_sampled_data = None
        self.last_sampled_timestamp = None
        self.last_sample_step_timestamp = None

    def resample(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        if self.last_sample_step_timestamp is None:
            return [self._initial_sample(data, relative_timestamp)]
        else:
            return self._samples_until(data, relative_timestamp)

    def _initial_sample(self, data: InputData, relative_timestamp: float) -> Sample[InputData]:
        self.last_sampled_data = data
        self.last_sampled_timestamp = relative_timestamp
        self.last_sample_step_timestamp = relative_timestamp

        return Sample(data=self.last_sampled_data, timestamp=self.last_sampled_timestamp)

    def _samples_until(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        assert (
            self.last_sampled_data is not None
            and self.last_sampled_timestamp is not None
            and self.last_sample_step_timestamp
        ), "There must have been an initial sample"

        if self.is_timestamp_after_next_sampling_step(relative_timestamp):
            self.last_sampled_data = data
            self.last_sampled_timestamp = relative_timestamp
            self.last_sample_step_timestamp = self.last_sample_step_timestamp + self.sampling_step_in_seconds
            return [Sample(data=self.last_sampled_data, timestamp=self.last_sampled_timestamp)]

        return [Sample(data=self.last_sampled_data, timestamp=self.last_sampled_timestamp, was_sampled_already=True)]

    def is_timestamp_after_next_sampling_step(self, relative_timestamp: float) -> bool:
        if self.last_sample_step_timestamp is None:
            # There was no previous sample, so it is time to sample
            return True

        return relative_timestamp - self.last_sample_step_timestamp >= self.sampling_step_in_seconds
