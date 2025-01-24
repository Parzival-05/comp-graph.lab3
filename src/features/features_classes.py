from dataclasses import dataclass


@dataclass
class Feature:
    """
    Abstract base class for all feature types.

    This class serves as a foundation for different types of features
    extracted from rice grain images.
    """

    pass


@dataclass
class MeasureFeature(Feature):
    """
    Represents measurement features of a rice grain.

    Attributes:
        width (int): The width of the rice grain in pixels.
        height (int): The height of the rice grain in pixels.
    """

    width: int
    height: int


@dataclass
class BrightnessFeature(Feature):
    """
    Represents the brightness feature of a rice grain.

    Attributes:
        average_brightness (float): The average brightness of the rice grain image.
    """

    average_brightness: float


@dataclass
class SizeFeature(Feature):
    """
    Represents the size feature of a rice grain.

    Attributes:
        area (float): The area of the rice grain.
    """

    area: float


@dataclass
class RiceFeatures(Feature):
    """
    Aggregates all feature types related to a rice grain.

    This class serves as a container for features extracted from a rice grain image.

    Attributes:
        measure (MeasureFeature): An instance containing width and height measurements.
        brightness (BrightnessFeature): An instance containing brightness information.
        size (SizeFeature): An instance containing area information.
    """

    measure: MeasureFeature
    brightness: BrightnessFeature
    size: SizeFeature
