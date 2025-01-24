from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.common import IMAGE_SHAPE
from src.infer import LABELS, infer, infer_by_path
from tests.constants import DUMMY_IMAGE_PATH


@patch("src.infer.model")  # mock the model object
def test_infer_valid_image(mocked_model):
    dummy_image = np.random.rand(*IMAGE_SHAPE).astype(np.float32)

    mocked_model.return_value = torch.tensor([[2.5, 1.0, 3.0, 0.5, 0.0]])

    probabilities = infer(dummy_image)

    assert len(probabilities) == len(LABELS), "Incorrect number of probabilities"
    assert sum(probabilities) == pytest.approx(
        1.0, rel=1e-3
    ), "Probabilities do not sum to 1"
    assert probabilities[2] == max(
        probabilities
    ), "Class Ipsala should have the highest probability"

    mocked_model.assert_called_once()


@patch("src.infer.load_image")
@patch("src.infer.model")
def test_infer_by_path(mocked_model, mocked_load_image):
    dummy_image = np.random.rand(*IMAGE_SHAPE).astype(np.float32)
    mocked_load_image.return_value = dummy_image

    mocked_model.return_value = torch.tensor([[1.0, 0.5, 2.0, 1.5, 0.0]])

    probabilities = infer_by_path(DUMMY_IMAGE_PATH)

    mocked_load_image.assert_called_once_with(DUMMY_IMAGE_PATH)
    mocked_model.assert_called_once()
    assert len(probabilities) == len(LABELS), "Incorrect probabilities length"
    assert sum(probabilities) == pytest.approx(1.0), "Probabilities do not sum to 1"
