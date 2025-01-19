import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from src.common import RiceImageType

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
    if image.shape != (250, 250, 3):
        raise ValueError(
            f"Upload a three-channel 250x250 image, {str(image.shape)} is not suitable"
        )
    image = image.astype(np.float32) / 255.0  # type: ignore
    return image.transpose((2, 0, 1))


def load_image(image_path: str | Path) -> RiceImageType:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # type: ignore
    return preprocess_image(image)


def infer_by_path(image_path: str | Path) -> list:
    return infer(load_image(image_path))


def infer(image: RiceImageType) -> list:
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    return probabilities.tolist()[0]
