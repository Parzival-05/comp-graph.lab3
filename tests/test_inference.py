import pytest
import torch
import numpy as np
from unittest.mock import patch
from src.infer import infer, infer_by_path, LABELS

@patch("src.infer.model")  # mock the model object
def test_infer_valid_image(mocked_model):
    # dummy input image-like tensor (3, 250, 250) normalized
    dummy_image = np.random.rand(3, 250, 250).astype(np.float32)

    # mocked model output (logits for 5 classes, corresponding to LABELS)
    mocked_model.return_value = torch.tensor([[2.5, 1.0, 3.0, 0.5, 0.0]])

    probabilities = infer(dummy_image)

    # check that the probabilities are a valid distribution
    assert len(probabilities) == len(LABELS), "Incorrect number of probabilities"
    assert sum(probabilities) == pytest.approx(1.0, rel=1e-3), "Probabilities do not sum to 1"
    assert probabilities[2] == max(probabilities), "Class Ipsala should have the highest probability"

    mocked_model.assert_called_once()

@patch("src.infer.model")
def test_infer_random_noise(mocked_model):
    random_noise_image = np.random.rand(3, 250, 250).astype(np.float32)
    # random logits
    mocked_model.return_value = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

    probabilities = infer(random_noise_image)
    assert probabilities == pytest.approx([0.2, 0.2, 0.2, 0.2, 0.2], rel=1e-3), "Probabilities should be uniform"

@patch("src.infer.load_image")
@patch("src.infer.model")
def test_infer_by_path(mocked_model, mocked_load_image):
    dummy_image = np.random.rand(3, 250, 250).astype(np.float32)
    mocked_load_image.return_value = dummy_image

    mocked_model.return_value = torch.tensor([[1.0, 0.5, 2.0, 1.5, 0.0]])

    probabilities = infer_by_path("dummy/image/path.jpg")

    mocked_load_image.assert_called_once_with("dummy/image/path.jpg")
    mocked_model.assert_called_once()
    assert len(probabilities) == len(LABELS), "Incorrect probabilities length"
    assert sum(probabilities) == pytest.approx(1.0), "Probabilities do not sum to 1"
