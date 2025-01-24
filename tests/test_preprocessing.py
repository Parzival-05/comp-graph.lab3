import numpy as np
import pytest

from src.common import IMAGE_SHAPE, PROCESSED_IMAGE_SHAPE
from src.infer import preprocess_image
from tests.constants import (
    INVALID_IMAGE_SHAPE,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
    PIXEL_MAX_VALUE,
    PIXEL_MIN_VALUE,
)


def test_preprocess_image_valid():
    image = np.random.randint(
        PIXEL_MIN_VALUE, PIXEL_MAX_VALUE, IMAGE_SHAPE, dtype=np.uint8
    )
    processed_image = preprocess_image(image)

    assert (
        processed_image.shape == PROCESSED_IMAGE_SHAPE
    ), "Processed image has incorrect shape"
    assert (
        NORMALIZED_MIN <= processed_image.min()
        and processed_image.max() <= NORMALIZED_MAX
    ), "Image not normalized"


def test_preprocess_image_invalid_size():
    image = np.random.randint(
        PIXEL_MIN_VALUE, PIXEL_MAX_VALUE, INVALID_IMAGE_SHAPE, dtype=np.uint8
    )

    with pytest.raises(ValueError, match="Upload a 3-channel 250x250 image"):
        preprocess_image(image)
