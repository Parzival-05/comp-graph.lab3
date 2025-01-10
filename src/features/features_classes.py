from dataclasses import dataclass


@dataclass
class Feature:
    pass


@dataclass
class MeasureFeature(Feature):
    weight: int
    height: int


@dataclass
class BrightnessFeature(Feature):
    average_brightness: float


@dataclass
class MeasureBrightnessFeature(Feature):
    measure: MeasureFeature
    brightness: BrightnessFeature



