"""utilities for torchvision *RCNN models"""
from typing import Dict, Tuple

import torch
from torchvision.transforms import ToTensor

from project.utils.ds.structures import BoxedExample


def to_model_input(
        example: BoxedExample
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    features = ToTensor()(example.image)
    labels = {
        "boxes": torch.tensor(example.boxes).float(),
        "labels": torch.tensor([0] * example.boxes.shape[0])
    }
    return features, labels
