from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

from src.common import RiceImageType
from src.features.features_classes import (
    BrightnessFeature,
    Feature,
    MeasureBrightnessFeature,
    MeasureFeature,
    SizeFeature,
)

T = TypeVar("T", bound=Feature)


class FeatureExtraction(ABC, Generic[T]):
    def __init__(self, extractor: Callable[[RiceImageType], T]):
        self._extractor = extractor

    @abstractmethod
    def _extract(self, image: RiceImageType) -> T: ...

    def extract(self, image: RiceImageType) -> T:
        return self._extractor(image)


class MeasureFeatureExtraction(FeatureExtraction[MeasureFeature]):
    def __init__(self):
        super().__init__(self._extract)

    def _extract(self, image: RiceImageType) -> MeasureFeature:
        return MeasureFeature(weight=3, height=4)


class BrightnessFeatureExtraction(FeatureExtraction[BrightnessFeature]):
    def __init__(self):
        super().__init__(self._extract)

    def _extract(self, image: RiceImageType) -> BrightnessFeature:
        return BrightnessFeature(average_brightness=30.0)


class SizeFeatureExtraction(FeatureExtraction[SizeFeature]):
    def __init__(self):
        super().__init__(self._extract)

    def _extract(self, image: RiceImageType) -> SizeFeature:
        return SizeFeature(area=30.0)


class MeasureBrightnessFeatureExtraction(FeatureExtraction[MeasureBrightnessFeature]):
    def __init__(
        self,
        measure_extractor: FeatureExtraction[MeasureFeature],
        brightness_extractor: FeatureExtraction[BrightnessFeature],
        size_extractor: FeatureExtraction[SizeFeature],
    ):
        super().__init__(self._extract)
        self.measure_extractor = measure_extractor
        self.brightness_extractor = brightness_extractor
        self.size_extractor = size_extractor

    def _extract(self, image: RiceImageType) -> MeasureBrightnessFeature:
        mf = self.measure_extractor.extract(image)
        bf = self.brightness_extractor.extract(image)
        sf = self.size_extractor.extract(image)
        return MeasureBrightnessFeature(measure=mf, brightness=bf, size=sf)
