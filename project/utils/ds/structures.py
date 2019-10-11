import dataclasses as dc
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL.ImageDraw import Draw

from project.utils.box_utils import box_from_single_segmask

_IMAGES_FOLDER_NAME = "images"
_MASKS_FOLDER_NAME = "masks"


@dc.dataclass(frozen=True)
class Example:
    image: Image.Image
    name: str = ""

    @classmethod
    def from_path(cls, path: Path):
        image_files = list((path / _IMAGES_FOLDER_NAME).iterdir())

        return cls(
            image=Image.open(image_files[0]),
            name=path.name
        )


@dc.dataclass(frozen=True)
class BoxedExample(Example):
    boxes: np.ndarray = np.empty((0, 4))

    def vis_boxes(self, thickness: int = 5) -> Image.Image:
        if self.boxes.shape[0] == 0:
            return self.image
        img = self.image.copy()
        drawer = Draw(img)

        for box in (self.boxes):
            drawer.rectangle(
                box.tolist()
            )
        return img


@dc.dataclass(frozen=True)
class MaskedExample(BoxedExample):
    masks: Optional[np.ndarray] = np.empty(0)

    @classmethod
    def from_path(cls, path: Path):
        obj: MaskedExample = super().from_path(path)
        masks = np.empty((0, obj.image.size[0], obj.image.size[1]))
        boxes = np.empty((0, 4))

        mask_path = path / _MASKS_FOLDER_NAME

        if mask_path.exists():
            masks = np.array([
                (imageio.imread(m) != 0).astype(int) for m in sorted(list(mask_path.iterdir()))
            ])
            boxes = np.array([
                box_from_single_segmask(m)
                for m in masks
            ])
        return dc.replace(
            obj,
            masks=masks,
            boxes=boxes
        )

    def vis_segmasks(self, alpha: float = 0.5):
        masks = self.masks.sum(axis=0)

        plt.imshow(self.image)
        plt.imshow(masks, alpha=alpha)
        plt.show()
