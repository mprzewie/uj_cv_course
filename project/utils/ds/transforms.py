from typing import Tuple, Callable

import torch
from torchvision.transforms import ToTensor

from project.utils.ds.structures import BoxedExample

BoxedExampleTransform = Callable[[BoxedExample], BoxedExample]


def convert_image_fn(mode: str) -> BoxedExampleTransform:
    def to_mode(example: BoxedExample) -> BoxedExample:
        return example.replace(image=example.image.convert(mode))

    return to_mode


to_rgb = convert_image_fn("RGB")
to_greyscale = convert_image_fn("LA")


def to_tensors_tuple(example: BoxedExample) -> Tuple[torch.Tensor, torch.Tensor]:
    return ToTensor()(example.image), torch.tensor(example.normed_boxes_x_y_w_h)


# TODO
# functions which return BoxedExample -> BoxedExample functions

def scale_to(width: int, height: int) -> BoxedExampleTransform:
    pass


def random_crop(width: int, height: int) -> BoxedExampleTransform:
    pass


def random_rotate_n_90(example: BoxedExample) -> BoxedExample:
    pass


def flip_color_on_intensity_heuristic(example: BoxedExample) -> BoxedExample:
    pass


def normalize_image(mean: float, std: float) -> BoxedExampleTransform:
    def normalize(example: BoxedExample) -> BoxedExample:
        pass

    return normalize