from soccer_diffusion.dataset.imports.data import InputData
from soccer_diffusion.dataset.resampling.resampler import Resampler, Sample


class OriginalRateResampler(Resampler):
    def resample(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        return [Sample(data=data, timestamp=relative_timestamp)]
