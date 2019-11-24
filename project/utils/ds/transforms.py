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


def flip_color_on_intensity_heuristic(example: BoxedExample) -> BoxedExample:
    rgb_example = to_rgb(example)

    intensity = np.sqrt((rgb_example.image_array ** 2).sum(axis=2))
    median_int = np.median(intensity, axis=(0, 1))
    mean_int = np.mean(intensity, axis=(0, 1))
    if mean_int < median_int:
        example = change_image_mode_fn(example.image.mode)(example.replace(
            image=ImageOps.invert(rgb_example.image)
        ))

    return example


def normalize_image(mean: float, std: float) -> BoxedExampleTransform:
    def normalize(example: BoxedExample) -> BoxedExample:
        pass

    return normalize
