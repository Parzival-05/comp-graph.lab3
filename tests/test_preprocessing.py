import numpy as np
import pytest
from src.infer import preprocess_image


def test_preprocess_image_valid():
    image = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    processed_image = preprocess_image(image)

    assert processed_image.shape == (3, 250, 250), "Processed image has incorrect shape"
    assert processed_image.max() <= 1.0 and processed_image.min() >= 0.0, "Image not normalized"


def test_preprocess_image_invalid_size():
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Upload a three-channel 250x250 image"):
        preprocess_image(image)