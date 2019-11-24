from typing import Callable, Any, Dict

import numpy as np

from project.utils.ds.structures import BoxedExample
from project.utils.ds.transforms import BoxedExampleTransform

SamplerFn = Callable[[], Dict]


def rotate_sampler(amplitude: float = 15) -> SamplerFn:
    return lambda: {
        "degree": np.random.uniform(-amplitude, amplitude)
    }


def rotate_90_sampler() -> SamplerFn:
    return lambda: {
        "degree": np.random.choice([0, 90, 180, 270])
    }


def resize_sampler(amplitude: float = 0.5):
    return lambda: {
        "x_factor": 1 + np.random.uniform(-amplitude, amplitude),
        "y_factor": 1 + np.random.uniform(-amplitude, amplitude)
    }


def constant_crop_size_sampler(crop_size: float = 0.7):
    def fn():
        x_min = np.random.uniform(0, 1 - crop_size)
        y_min = np.random.uniform(0, 1 - crop_size)
        return {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_min + crop_size,
            "y_max": y_min + crop_size
        }
    return fn


def random_transform(
        transform_fn: Callable[[BoxedExample, Any], BoxedExample],
        sampler: SamplerFn = lambda: dict()
) -> BoxedExampleTransform:
    def fn(ex: BoxedExample) -> BoxedExample:
        kwargs = sampler()
        return transform_fn(ex, **kwargs)

    return fn
