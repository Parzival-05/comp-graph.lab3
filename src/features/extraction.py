from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

import cv2
import numpy as np

from src.common import RiceImageType
from src.features.extraction_utils import get_contours, get_largest_contour
from src.features.features_classes import (
    BrightnessFeature,
    Feature,
    MeasureFeature,
    RiceFeatures,
    SizeFeature,
)

T = TypeVar("T", bound=Feature)


class FeatureExtraction(ABC, Generic[T]):
    """
    Abstract base class for feature extraction from images.

    This class defines a generic interface for extracting features of type T from an image.
    It uses a provided extractor function to perform the actual extraction.

    Args:
        extractor: A function that takes an image and returns a feature of type T.
    """

    def __init__(self, extractor: Optional[Callable[[RiceImageType], T]] = None):
        if extractor is None:
            extractor = self._extract
        self._extractor = extractor

    @abstractmethod
    def _extract(self, image: RiceImageType) -> T:
        """
        Abstract method to perform feature extraction.

        Subclasses must implement this method to define their specific extraction logic.

        Args:
            image: The input image.

        Returns:
            T: The extracted feature.
        """
        ...

    def extract(self, image: RiceImageType) -> T:
        """
        Extracts a feature from the given image using the provided extractor function.

        Args:
            image (RiceImageType): The input image.

        Returns:
            T: The extracted feature.
        """
        return self._extractor(image)


class MeasureFeatureExtraction(FeatureExtraction[MeasureFeature]):
    def _extract(self, image: RiceImageType) -> MeasureFeature:
        """
        Steps:
            1. Retrieve the rice contour by `extraction_utils.get_largest_contour`.
            2. Calculate the angle to determine orientation.
            3. Correct the angle and rotate the image to align the rice grain horizontally.
            4. Retrieve the rice contour from the rotated image.
            5. Calculate the bounding rectangle dimensions and return the width and height of the rice grain.
        """
        contours = get_contours(image)
        if not contours:
            raise RuntimeError("There is no rice on image")
        rice_contour = get_largest_contour(contours)

        rect = cv2.minAreaRect(rice_contour)
        # Calculate the angle to determine orientation
        angle = rect[2]

        if angle < -45:
            angle += 90

        center = tuple(map(int, rect[0]))

        # Correct the angle and rotate the image to align the rice grain horizontally
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC
        )

        contours_rotated = get_contours(rotated)
        if not contours_rotated:
            raise RuntimeError("There is no rice on image")
        new_contour = get_largest_contour(contours_rotated)

        # Calculate the bounding rectangle dimensions and return the width and height of the rice grain
        _, _, w, h = cv2.boundingRect(new_contour)

        height = max(w, h)
        width = min(w, h)
        return MeasureFeature(width=width, height=height)


class BrightnessFeatureExtraction(FeatureExtraction[BrightnessFeature]):
    def _extract(self, image: RiceImageType) -> BrightnessFeature:
        """
        Steps:
            1. Retrieve the rice contour by `extraction_utils.get_largest_contour`.
            2. Convert the original image to HSV color space and extract the Value (brightness) channel.
            3. Create a mask based on the rice grain contour.
            4. Extract pixel values from the Value channel that correspond to the rice grain using the mask.
            5. Calculate the average brightness of the extracted rice grain pixels.
        """
        contours = get_contours(image)
        if not contours:
            return BrightnessFeature(average_brightness=0.0)
        rice_contour = get_largest_contour(contours)

        # Convert the image to HSV and extract the Value channel
        V = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]

        # Create a mask based on the rice grain contour
        mask = np.zeros_like(V, dtype=np.uint8)
        cv2.drawContours(mask, [rice_contour], -1, 255, thickness=cv2.FILLED)  # type: ignore

        # Extract pixel values of the rice grain using the mask
        rice_pixels = V[mask == 255]

        # Calculate the average brightness
        average_brightness = float(np.mean(rice_pixels))
        return BrightnessFeature(average_brightness=average_brightness)


class SizeFeatureExtraction(FeatureExtraction[SizeFeature]):
    def _extract(self, image: RiceImageType) -> SizeFeature:
        """
        Steps:
            1. Retrieve the rice contour by `extraction_utils.get_largest_contour`.
            2. Calculate the area of the largest contour.
            3. Return the area as a SizeFeature instance.
        """
        contours = get_contours(image)
        if not contours:
            return SizeFeature(area=0.0)

        rice_contour = get_largest_contour(contours)
        rice_area = cv2.contourArea(rice_contour)
        return SizeFeature(area=rice_area)


class RiceFeaturesExtraction(FeatureExtraction[RiceFeatures]):
    def __init__(
        self,
        measure_extractor: FeatureExtraction[MeasureFeature],
        brightness_extractor: FeatureExtraction[BrightnessFeature],
        size_extractor: FeatureExtraction[SizeFeature],
    ):
        super().__init__()
        self.measure_extractor = measure_extractor
        self.brightness_extractor = brightness_extractor
        self.size_extractor = size_extractor

    def _extract(self, image: RiceImageType) -> RiceFeatures:
        mf = self.measure_extractor.extract(image)
        bf = self.brightness_extractor.extract(image)
        sf = self.size_extractor.extract(image)
        return RiceFeatures(measure=mf, brightness=bf, size=sf)
