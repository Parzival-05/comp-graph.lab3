import pytest
import numpy as np
from src.features.extraction import MeasureFeatureExtraction, BrightnessFeatureExtraction

# dummy image with a large white rectangle (simplified rice grain)
def test_measure_feature_extraction():
    import cv2
    extractor = MeasureFeatureExtraction()
    image = np.zeros((250, 250, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (200, 200), (255, 255, 255), -1)

    feature = extractor.extract(image)
    assert feature.width > 0 and feature.height > 0, "Width or height is not computed correctly"

# dummy image with varying brightness
def test_brightness_feature_extraction():
    extractor = BrightnessFeatureExtraction()
    image = np.ones((250, 250, 3), dtype=np.uint8) * 128  # mid-brightness
    feature = extractor.extract(image)
    assert feature.average_brightness == pytest.approx(128, rel=1e-2)