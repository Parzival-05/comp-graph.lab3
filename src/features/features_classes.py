from dataclasses import dataclass


@dataclass
class Feature:
    pass


@dataclass
class MeasureFeature(Feature):
    width: int
    height: int


@dataclass
class BrightnessFeature(Feature):
    average_brightness: float


@dataclass
class SizeFeature(Feature):
    area: float


@dataclass
class RiceFeatures(Feature):
    measure: MeasureFeature
    brightness: BrightnessFeature
    size: SizeFeature
