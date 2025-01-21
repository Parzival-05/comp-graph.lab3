import pytest
import numpy as np
import cv2
from src.features.extraction import (
    MeasureFeatureExtraction,
    BrightnessFeatureExtraction,
    SizeFeatureExtraction,
    RiceFeaturesExtraction,
)
from src.features.features_classes import MeasureFeature, BrightnessFeature, SizeFeature, RiceFeatures


def create_dummy_image(size=(250, 250), color=(255, 255, 255)) -> np.ndarray:
    """
    Creates a dummy image with a solid color (default white).
    """
    return np.full((*size, 3), color, dtype=np.uint8)


def create_dummy_image_with_rice(size=(250, 250)) -> np.ndarray:
    """
    Creates a dummy image with a "rice grain" shape (rectangular white region on a black background).
    """
    image = np.zeros((*size, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 200), (255, 255, 255), -1)
    return image


### MeasureFeatureExtraction

def test_measure_feature_valid_rice():
    extractor = MeasureFeatureExtraction()
    dummy_image = create_dummy_image_with_rice()

    feature = extractor.extract(dummy_image)
    assert isinstance(feature, MeasureFeature), "Returned feature is not of type MeasureFeature"
    assert feature.width > 0 and feature.height > 0, "Width or height should be greater than 0"


def test_measure_feature_no_rice():
    extractor = MeasureFeatureExtraction()
    dummy_image = create_dummy_image(color=(0, 0, 0))  # Empty image (black background)

    with pytest.raises(RuntimeError, match="There is no rice on image"):
        extractor.extract(dummy_image)


def test_measure_feature_rotated_rice():
    extractor = MeasureFeatureExtraction()
    dummy_image = create_dummy_image_with_rice()

    # rotate the image by placing the "rice" at an angle
    rows, cols = dummy_image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), 45, 1)
    rotated_image = cv2.warpAffine(dummy_image, M, (cols, rows))

    feature = extractor.extract(rotated_image)
    assert feature.width > 0 and feature.height > 0, "Width or height should be greater than 0"


### BrightnessFeatureExtraction

def test_brightness_feature_valid_rice():
    extractor = BrightnessFeatureExtraction()
    dummy_image = create_dummy_image_with_rice()

    feature = extractor.extract(dummy_image)
    assert isinstance(feature, BrightnessFeature), "Returned feature is not of type BrightnessFeature"
    assert 0 <= feature.average_brightness <= 255, "Average brightness should be in the range [0, 255]"


def test_brightness_feature_no_rice():
    extractor = BrightnessFeatureExtraction()
    dummy_image = create_dummy_image(color=(0, 0, 0))  # empty black image

    feature = extractor.extract(dummy_image)
    assert isinstance(feature, BrightnessFeature), "Returned feature is not of type BrightnessFeature"
    assert feature.average_brightness == 0.0, "Average brightness should be 0 for images with no rice"

def test_brightness_feature_mild_rice_with_contours():
    extractor = BrightnessFeatureExtraction()

    image = np.ones((250, 250, 3), dtype=np.uint8) * 128  # mid-brightness background
    cv2.rectangle(image, (80, 80), (170, 170), (200, 200, 200), -1)  # brighter "rice grain"

    feature = extractor.extract(image)

    assert feature.average_brightness == pytest.approx(137.0, rel=1e-1), "Average brightness of rice is not accurate"

### SizeFeatureExtraction

def test_size_feature_valid_rice():
    extractor = SizeFeatureExtraction()
    dummy_image = create_dummy_image_with_rice()

    feature = extractor.extract(dummy_image)
    assert isinstance(feature, SizeFeature), "Returned feature is not of type SizeFeature"
    assert feature.area > 0, "Area should be greater than 0 for valid rice"


def test_size_feature_no_rice():
    extractor = SizeFeatureExtraction()
    dummy_image = create_dummy_image(color=(0, 0, 0))  # Empty image (black background)

    feature = extractor.extract(dummy_image)
    assert isinstance(feature, SizeFeature), "Returned feature is not of type SizeFeature"
    assert feature.area == 0.0, "Area should be 0 for an empty image"


### RiceFeaturesExtraction

def test_combined_feature_extraction():
    dummy_image = create_dummy_image_with_rice()
    measure_extractor = MeasureFeatureExtraction()
    brightness_extractor = BrightnessFeatureExtraction()
    size_extractor = SizeFeatureExtraction()

    extractor = RiceFeaturesExtraction(measure_extractor, brightness_extractor, size_extractor)
    features = extractor.extract(dummy_image)

    # assertions for combined features
    assert isinstance(features, RiceFeatures), "Returned features is not of type RiceFeatures"
    assert isinstance(features.measure, MeasureFeature), "Measure feature is not extracted correctly"
    assert isinstance(features.brightness, BrightnessFeature), "Brightness feature is not extracted correctly"
    assert isinstance(features.size, SizeFeature), "Size feature is not extracted correctly"

    # additional validations
    assert features.measure.width > 0 and features.measure.height > 0, "Width and height should be greater than 0"
    assert 0 <= features.brightness.average_brightness <= 255, "Brightness must be in the range [0, 255]"
    assert features.size.area > 0, "Area must be greater than 0 for a valid image with rice"


### edge cases
def test_combined_no_rice():
    dummy_image = create_dummy_image(color=(0, 0, 0))  # empty black image
    measure_extractor = MeasureFeatureExtraction()
    brightness_extractor = BrightnessFeatureExtraction()
    size_extractor = SizeFeatureExtraction()

    extractor = RiceFeaturesExtraction(measure_extractor, brightness_extractor, size_extractor)

    # MeasureFeatureExtraction should raise RuntimeError
    with pytest.raises(RuntimeError, match="There is no rice on image"):
        extractor.extract(dummy_image)