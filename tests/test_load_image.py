import pytest
from src.infer import load_image


def test_load_image_valid(tmp_path):
    import cv2
    import numpy as np
    image = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), image)

    processed_image = load_image(img_path)
    assert processed_image.shape == (3, 250, 250), "Processed image has incorrect shape"


def test_load_image_invalid_path():
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_file.jpg")