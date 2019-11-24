from typing import Tuple, Callable

import PIL
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor

from project.utils.ds.boxes import areas
from project.utils.ds.structures import BoxedExample

BoxedExampleTransform = Callable[[BoxedExample], BoxedExample]


def change_image_mode_fn(mode: str) -> BoxedExampleTransform:
    def to_mode(example: BoxedExample) -> BoxedExample:
        return example.replace(image=example.image.convert(mode))

    return to_mode


to_rgb = change_image_mode_fn("RGB")
to_grayscale = change_image_mode_fn("LA")


def to_tensors_tuple(example: BoxedExample) -> Tuple[torch.Tensor, torch.Tensor]:
    return ToTensor()(example.image), torch.tensor(example.normed_boxes_x_y_w_h)


def keep_mode(transform_fn: BoxedExampleTransform) -> BoxedExampleTransform:
    """
    A decorator which ensures that image mode is not changed during a transformation
    """
    def _fn(example: BoxedExample, *args, **kwargs) -> BoxedExample:
        mode = example.image.mode
        return change_image_mode_fn(mode)(
            transform_fn(example, *args, **kwargs)
        )

    return _fn


@keep_mode
def resize(example: BoxedExample, x_factor: float = 1, y_factor: float = 1) -> BoxedExample:
    target_x = int(example.image.width * x_factor)
    target_y = int(example.image.height * y_factor)

    resize_vector = np.array([x_factor, y_factor, x_factor, y_factor])
    return example.replace(
        image=example.image.resize(
            (target_x, target_y),
            Image.BILINEAR
        ),
        boxes=example.boxes * resize_vector
    )


@keep_mode
def crop(
        example: BoxedExample,
        x_min: float = 0, y_min: float = 0, x_max: float = 1, y_max: float = 1
) -> BoxedExample:
    """Normed coordinates"""
    abs_x_min, abs_x_max = [
        int(example.image.width * x) for x in [x_min, x_max]
    ]

    abs_y_min, abs_y_max = [
        int(example.image.height * y) for y in [y_min, y_max]
    ]

    cropped_boxes = np.array([
        [
            min(max(xmin, abs_x_min), abs_x_max),
            min(max(ymin, abs_y_min), abs_y_max),
            min(max(xmax, abs_x_min), abs_x_max),
            min(max(ymax, abs_y_min), abs_y_max),
        ]
        for [xmin, ymin, xmax, ymax]
        in example.boxes
    ]) - np.array([
        abs_x_min, abs_y_min, abs_x_min, abs_y_min
    ])

    cropped_boxes = cropped_boxes[areas(cropped_boxes) > 0]

    return example.replace(
        image=example.image.crop(box=(abs_x_min, abs_y_min, abs_x_max, abs_y_max)),
        boxes=cropped_boxes
    )


@keep_mode
def rotate(example: BoxedExample, degree=0) -> BoxedExample:
    angle = - 2 * np.pi * degree / 360
    rot_mat = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    box_rot_mat = np.zeros((4, 4))
    box_rot_mat[0:2, 0:2] = rot_mat
    box_rot_mat[2:4, 2:4] = rot_mat

    center_vector = np.array([
        example.image.width,
        example.image.height,
        example.image.width,
        example.image.height
    ]) / 2
    centered_boxes = example.boxes - center_vector
    rot_boxes = (box_rot_mat @ centered_boxes.T).T + center_vector
    return example.replace(
        image=example.image.rotate(degree),
        boxes=rot_boxes
    )


@keep_mode
def flip_color_on_intensity_heuristic(example: BoxedExample) -> BoxedExample:
    rgb_example_array_normed = to_rgb(example).image_array / 255
    intensity = np.sqrt((rgb_example_array_normed ** 2).sum(axis=2))
    median_int = np.median(intensity)
    mean_int = np.mean(intensity)
    if mean_int < median_int:
        example = example.replace(
            image=ImageOps.invert(to_rgb(example).image)
        )

    return example


@keep_mode
def make_max_255(example: BoxedExample) -> BoxedExample:
    rgb_example = to_rgb(example)
    max_pixel = rgb_example.image_array.max()
    factor = 255 / max_pixel
    factored = rgb_example.image_array * factor
    return example.replace(image=Image.fromarray(factored.astype("uint8")))


# TODO random transformation pipeline
