from typing import TypeAlias

from nptyping import Int, NDArray, Shape

IMAGE_HEIGHT = 250
IMAGE_WIDTH = 250
IMAGE_CHANNELS = 3
# There is only one image shape from the Rice image dataset
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
PROCESSED_IMAGE_SHAPE = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

RiceImageType: TypeAlias = NDArray[
    Shape[f"{IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS}"], Int
]
