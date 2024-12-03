from ddlitlab2024.dataset.imports.data import InputData
from ddlitlab2024.dataset.resampling.resampler import Resampler, Sample


class OriginalRateResampler(Resampler):
    def resample(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        return [Sample(data=data, timestamp=relative_timestamp)]
