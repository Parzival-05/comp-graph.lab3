from typing import Sequence

import cv2
from cv2.typing import MatLike


def get_contours(image: MatLike, grayscale_threshold=50) -> Sequence[MatLike]:
    """
    Retrieves contours from an image of a rice grain on a black background.

    This function processes the input image to identify and extract contours
    representing the rice grain. It assumes that the image contains a single
    rice grain against a predominantly black background.

    Steps:
        1. Convert the input image from BGR to grayscale.
        2. Apply binary thresholding to separate the rice grain from the background.
        3. Find and return contours in the thresholded image.

    Args:
        image (MatLike): The input image containing a single rice grain on a black background.

    Returns:
        Sequence[MatLike]: A list of contours found in the image. Each contour is represented
                           as a NumPy array of (x, y) coordinates.

    Raises:
        ValueError: If the input image is not a valid NumPy array or does not have three color channels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, grayscale_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_largest_contour(contours: Sequence[MatLike]) -> MatLike:
    """
    Retrieves the largest contour from a list of contours based on area.
    """
    return max(contours, key=cv2.contourArea)
