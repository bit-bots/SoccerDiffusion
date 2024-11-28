from ddlitlab2024.dataset.imports.model_importer import InputData, Sample


class OriginalRateResampler:
    def resample(self, data: InputData, relative_timestamp: float) -> list[Sample[InputData]]:
        return [Sample(data=data, timestamp=relative_timestamp)]
