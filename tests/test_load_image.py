import pytest

from src.common import IMAGE_SHAPE, PROCESSED_IMAGE_SHAPE
from src.infer import load_image
from tests.constants import (
    NONEXISTENT_PATH,
    TEST_IMAGE_NAME,
)


def test_load_image_valid(tmp_path):
    import cv2
    import numpy as np

    image = np.random.randint(0, 255, IMAGE_SHAPE, dtype=np.uint8)
    img_path = tmp_path / TEST_IMAGE_NAME
    cv2.imwrite(str(img_path), image)

    processed_image = load_image(img_path)
    assert (
        processed_image.shape == PROCESSED_IMAGE_SHAPE
    ), "Processed image has incorrect shape"


def test_load_image_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_image(NONEXISTENT_PATH)
