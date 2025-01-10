from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

import cv2
import numpy as np

from src.common import RiceImageType
from src.features.features_classes import (
    BrightnessFeature,
    Feature,
    MeasureFeature,
    RiceFeatures,
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise RuntimeError("There is no rice on image")

        largest_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]

        if angle < -45:
            angle += 90

        center = tuple(map(int, rect[0]))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC
        )

        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, thresh_rotated = cv2.threshold(gray_rotated, 50, 255, cv2.THRESH_BINARY)
        contours_rotated, _ = cv2.findContours(
            thresh_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours_rotated:
            raise RuntimeError("There is no rice on image")

        new_contour = max(contours_rotated, key=cv2.contourArea)
        _, _, w, h = cv2.boundingRect(new_contour)

        height = max(w, h)
        width = min(w, h)
        return MeasureFeature(width=width, height=height)


class BrightnessFeatureExtraction(FeatureExtraction[BrightnessFeature]):
    def __init__(self):
        super().__init__(self._extract)

    def _extract(self, image: RiceImageType) -> BrightnessFeature:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return BrightnessFeature(average_brightness=0.0)
        rice_contour = max(contours, key=cv2.contourArea)

        V = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        mask = np.zeros_like(V, dtype=np.uint8)
        cv2.drawContours(mask, [rice_contour], -1, 255, thickness=cv2.FILLED)  # type: ignore

        rice_pixels = V[mask == 255]
        average_brightness = float(np.mean(rice_pixels))
        return BrightnessFeature(average_brightness=average_brightness)


class SizeFeatureExtraction(FeatureExtraction[SizeFeature]):
    def __init__(self):
        super().__init__(self._extract)

    def _extract(self, image: RiceImageType) -> SizeFeature:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return SizeFeature(area=0.0)
        rice_contour = max(contours, key=cv2.contourArea)

        rice_area = cv2.contourArea(rice_contour)
        return SizeFeature(area=rice_area)


class MeasureBrightnessFeatureExtraction(FeatureExtraction[RiceFeatures]):
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

    def _extract(self, image: RiceImageType) -> RiceFeatures:
        mf = self.measure_extractor.extract(image)
        bf = self.brightness_extractor.extract(image)
        sf = self.size_extractor.extract(image)
        return RiceFeatures(measure=mf, brightness=bf, size=sf)
