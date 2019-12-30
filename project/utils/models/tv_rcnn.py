"""utilities for torchvision *RCNN models"""
from typing import Dict, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import ToTensor, Normalize
from tqdm import tqdm

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


def evaluate_for_losses(
        model: FasterRCNN,
        data_loader: DataLoader,
        device: torch.device,
        max_n_batches: int = 1
) -> Dict[str, float]:
    model = model.to(device)
    model.train()
    loss_dicts = []
    with torch.no_grad():
        for i, (img, labels) in enumerate(tqdm(data_loader)):
            if i >= max_n_batches:
                break
            img = list(image.to(device) for image in img)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            loss_dicts.append(model(img, labels))

    return {
        k: np.mean([
            ld[k].cpu().item()
            for ld in loss_dicts
        ])
        for k in loss_dicts[0].keys()
    }


def coco_eval_metrics(coco_eval_obj: "CocoEvaluator") -> Dict[str, float]:
    """
    Summarization object
    https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/
    """
    metrics_names = [
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
        "Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] ",
    ]

    return {
        mn: coco_eval_obj.coco_eval["bbox"].stats[i]
        for (i, mn) in enumerate(metrics_names)
    }


def model_predictions_from_loader(
        model: FasterRCNN, data_loader: DataLoader, device: torch.device, max_n_batches=1,
        normalization_transform: Normalize = Normalize([1, 1, 1], [1, 1, 1])
) -> List[Tuple[BoxedExample, BoxedExample]]:
    model = model.to(device).eval()
    input_examples = []
    output_examples = []
    with torch.no_grad():
        for i, (img, labels) in enumerate(data_loader):
            if i >= max_n_batches:
                break
            outputs = model([i.to(device) for i in img])

            input_exs = [
                model_input_to_boxed_example(
                    (im, l),
                    normalization_transform
                )
                for (im, l) in zip(img, labels)
            ]
            output_exs = [
                i.replace(boxes=o["boxes"].cpu().numpy())

                for (i, o) in zip(input_exs, outputs)
            ]

            input_examples.extend(input_exs)
            output_examples.extend(output_exs)

    return list(zip(input_examples, output_examples))


