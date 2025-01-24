import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common import (
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_SHAPE,
    IMAGE_WIDTH,
    RiceImageType,
)

warnings.simplefilter("ignore")

PATH_TO_MODEL = "model/rice_classification_model.pt"
LABELS = {
    0: "Arborio",
    1: "Basmati",
    2: "Ipsala",
    3: "Jasmine",
    4: "Karacadag",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(PATH_TO_MODEL, map_location=torch.device(DEVICE))
model.eval()


def preprocess_image(image: np.ndarray) -> RiceImageType:
    """
    Preprocesses the input image for inference.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        RiceImageType: The preprocessed image as a NumPy array with shape (channels, height, width).

    Raises:
        ValueError: If the input image shape is not (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH).
    """
    if image.shape != IMAGE_SHAPE:
        raise ValueError(
            f"Upload a {IMAGE_CHANNELS}-channel {IMAGE_HEIGHT}x{IMAGE_WIDTH} image, {str(image.shape)} is not suitable"
        )
    image = image.astype(np.float32) / 255.0
    return image.transpose((2, 0, 1))


def load_image(image_path: str | Path) -> RiceImageType:
    """
    Loads and preprocesses an image from the specified path.

    Args:
        image_path (str | Path): The path to the image file.

    Returns:
        RiceImageType: The preprocessed image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file cannot be loaded from the specified path.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # type: ignore
    if image is None:
        raise FileNotFoundError(f"Could not load image from path {image_path}")
    return preprocess_image(image)


def infer_by_path(image_path: str | Path) -> list:
    """
    Performs inference on an image specified by its path.

    Args:
        image_path (str | Path): The path to the image file.

    Returns:
        list: A list of probabilities corresponding to each rice type.
    """
    return infer(load_image(image_path))


def infer(image: RiceImageType) -> list:
    """
    Performs inference on a preprocessed image.

    Args:
        image (RiceImageType): The preprocessed image as a NumPy array.

    Returns:
        list: A list of probabilities corresponding to each rice type.
    """
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    return probabilities.tolist()[0]
