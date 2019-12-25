"""utilities for torchvision *RCNN models"""
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Normalize

from project.utils.ds.boxes import areas
from project.utils.ds.structures import BoxedExample


def to_model_input(
        example: BoxedExample
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    features = ToTensor()(example.image)
    labels = {
        "boxes": torch.tensor(example.boxes).float(),
        "labels": torch.tensor([1] * example.boxes.shape[0]),  # PyTorch seems to think 0 is background
        "image_id": torch.tensor(example.id),
        "area": torch.tensor(areas(example.boxes)),
        "iscrowd": torch.tensor([0] * example.boxes.shape[0])
    }
    return features, labels


def model_input_to_boxed_example(
        model_input: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        normalization_transform: Normalize = Normalize([1, 1, 1], [1, 1, 1])
) -> BoxedExample:
    img_tensor, labels = model_input
    denormalized_img = (img_tensor.permute(1, 2, 0) * torch.tensor(normalization_transform.std)) + torch.tensor(
        normalization_transform.mean)
    return BoxedExample(
        image=Image.fromarray((denormalized_img.numpy() * 255).astype(np.uint8)),
        id=int(labels["image_id"]),
        boxes=labels["boxes"].numpy()
    )
