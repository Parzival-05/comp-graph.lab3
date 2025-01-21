import numpy as np
import pytest
from src.infer import preprocess_image
from tests.constants import (
    IMAGE_SHAPE,
    INVALID_IMAGE_SHAPE,
    PIXEL_MAX_VALUE,
    PIXEL_MIN_VALUE,
    NORMALIZED_MIN,
    NORMALIZED_MAX,
)


def test_preprocess_image_valid():
    image = np.random.randint(
        PIXEL_MIN_VALUE, PIXEL_MAX_VALUE, IMAGE_SHAPE, dtype=np.uint8
    )
    processed_image = preprocess_image(image)

    assert processed_image.shape == (3, 250, 250), "Processed image has incorrect shape"
    assert (
        NORMALIZED_MIN <= processed_image.min()
        and processed_image.max() <= NORMALIZED_MAX
    ), "Image not normalized"


def test_preprocess_image_invalid_size():
    image = np.random.randint(
        PIXEL_MIN_VALUE, PIXEL_MAX_VALUE, INVALID_IMAGE_SHAPE, dtype=np.uint8
    )

    with pytest.raises(ValueError, match="Upload a three-channel 250x250 image"):
        preprocess_image(image)
