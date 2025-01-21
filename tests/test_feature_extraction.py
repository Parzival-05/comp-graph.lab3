import pytest
import numpy as np
import cv2
from src.features.extraction import (
    MeasureFeatureExtraction,
    BrightnessFeatureExtraction,
    SizeFeatureExtraction,
    RiceFeaturesExtraction,
)
from src.features.features_classes import (
    MeasureFeature,
    BrightnessFeature,
    SizeFeature,
    RiceFeatures,
)
from tests.constants import (
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    RICE_RECT_TOP_LEFT,
    RICE_RECT_BOTTOM_RIGHT,
    BRIGHTNESS_MIN,
    BRIGHTNESS_MAX,
    RICE_BRIGHTNESS,
    MID_BRIGHTNESS,
    EXPECTED_BRIGHTNESS_APPROX,
    ROTATION_ANGLE,
)


def create_dummy_image(size=IMAGE_SIZE, color=(255, 255, 255)) -> np.ndarray:
    return np.full((*size, IMAGE_CHANNELS), color, dtype=np.uint8)


def create_dummy_image_with_rice(size=IMAGE_SIZE) -> np.ndarray:
    image = np.zeros((*size, IMAGE_CHANNELS), dtype=np.uint8)
    cv2.rectangle(
        image, RICE_RECT_TOP_LEFT, RICE_RECT_BOTTOM_RIGHT, (255, 255, 255), -1
    )
    return image


class TestMeasureFeatureExtraction:
    def test_measure_feature_valid_rice(self):
        extractor = MeasureFeatureExtraction()
        dummy_image = create_dummy_image_with_rice()

        feature = extractor.extract(dummy_image)
        assert isinstance(
            feature, MeasureFeature
        ), "Returned feature is not of type MeasureFeature"
        assert (
            feature.width > 0 and feature.height > 0
        ), "Width or height should be greater than 0"

    def test_measure_feature_no_rice(self):
        extractor = MeasureFeatureExtraction()
        dummy_image = create_dummy_image(
            color=(0, 0, 0)
        )  # Empty image (black background)

        with pytest.raises(RuntimeError, match="There is no rice on image"):
            extractor.extract(dummy_image)

    def test_measure_feature_rotated_rice(self):
        extractor = MeasureFeatureExtraction()
        dummy_image = create_dummy_image_with_rice()

        rows, cols = dummy_image.shape[:2]
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), ROTATION_ANGLE, 1)
        rotated_image = cv2.warpAffine(dummy_image, M, (cols, rows))

        feature = extractor.extract(rotated_image)
        assert (
            feature.width > 0 and feature.height > 0
        ), "Width or height should be greater than 0"


class TestBrightnessFeatureExtraction:
    def test_brightness_feature_valid_rice(self):
        extractor = BrightnessFeatureExtraction()
        dummy_image = create_dummy_image_with_rice()

        feature = extractor.extract(dummy_image)
        assert isinstance(
            feature, BrightnessFeature
        ), "Returned feature is not of type BrightnessFeature"
        assert (
            BRIGHTNESS_MIN <= feature.average_brightness <= BRIGHTNESS_MAX
        ), "Average brightness should be in the range [0, 255]"

    def test_brightness_feature_no_rice(self):
        extractor = BrightnessFeatureExtraction()
        dummy_image = create_dummy_image(color=(0, 0, 0))  # empty black image

        feature = extractor.extract(dummy_image)
        assert isinstance(
            feature, BrightnessFeature
        ), "Returned feature is not of type BrightnessFeature"
        assert (
            feature.average_brightness == 0.0
        ), "Average brightness should be 0 for images with no rice"

    def test_brightness_feature_mild_rice_with_contours(self):
        extractor = BrightnessFeatureExtraction()

        image = (
            np.ones((*IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.uint8) * MID_BRIGHTNESS
        )  # mid-brightness background
        cv2.rectangle(
            image,
            (80, 80),
            (170, 170),
            (RICE_BRIGHTNESS, RICE_BRIGHTNESS, RICE_BRIGHTNESS),
            -1,
        )  # brighter "rice grain"

        feature = extractor.extract(image)

        assert feature.average_brightness == pytest.approx(
            EXPECTED_BRIGHTNESS_APPROX, rel=1e-1
        ), "Average brightness of rice is not accurate"


class TestSizeFeatureExtraction:
    def test_size_feature_valid_rice(self):
        extractor = SizeFeatureExtraction()
        dummy_image = create_dummy_image_with_rice()

        feature = extractor.extract(dummy_image)
        assert isinstance(
            feature, SizeFeature
        ), "Returned feature is not of type SizeFeature"
        assert feature.area > 0, "Area should be greater than 0 for valid rice"

    def test_size_feature_no_rice(self):
        extractor = SizeFeatureExtraction()
        dummy_image = create_dummy_image(
            color=(0, 0, 0)
        )  # Empty image (black background)

        feature = extractor.extract(dummy_image)
        assert isinstance(
            feature, SizeFeature
        ), "Returned feature is not of type SizeFeature"
        assert feature.area == 0.0, "Area should be 0 for an empty image"


class TestRiceFeaturesExtraction:
    def test_combined_feature_extraction(self):
        dummy_image = create_dummy_image_with_rice()
        measure_extractor = MeasureFeatureExtraction()
        brightness_extractor = BrightnessFeatureExtraction()
        size_extractor = SizeFeatureExtraction()

        extractor = RiceFeaturesExtraction(
            measure_extractor, brightness_extractor, size_extractor
        )
        features = extractor.extract(dummy_image)

        assert isinstance(
            features, RiceFeatures
        ), "Returned features is not of type RiceFeatures"
        assert isinstance(
            features.measure, MeasureFeature
        ), "Measure feature is not extracted correctly"
        assert isinstance(
            features.brightness, BrightnessFeature
        ), "Brightness feature is not extracted correctly"
        assert isinstance(
            features.size, SizeFeature
        ), "Size feature is not extracted correctly"

        assert (
            features.measure.width > 0 and features.measure.height > 0
        ), "Width and height should be greater than 0"
        assert (
            BRIGHTNESS_MIN <= features.brightness.average_brightness <= BRIGHTNESS_MAX
        ), "Brightness must be in the range [0, 255]"
        assert (
            features.size.area > 0
        ), "Area must be greater than 0 for a valid image with rice"

    def test_combined_no_rice(self):
        dummy_image = create_dummy_image(color=(0, 0, 0))  # empty black image
        measure_extractor = MeasureFeatureExtraction()
        brightness_extractor = BrightnessFeatureExtraction()
        size_extractor = SizeFeatureExtraction()

        extractor = RiceFeaturesExtraction(
            measure_extractor, brightness_extractor, size_extractor
        )

        with pytest.raises(RuntimeError, match="There is no rice on image"):
            extractor.extract(dummy_image)
