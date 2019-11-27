import dataclasses as dc
from pathlib import Path
from typing import Optional, List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw

from project.utils.ds.boxes import box_from_single_segmask

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

    def replace(self, **kwargs) -> "Example":
        return dc.replace(self, **kwargs)

    @property
    def image_array(self) -> np.ndarray:
        return np.asarray(self.image)


@dc.dataclass(frozen=True)
class BoxedExample(Example):
    boxes: np.ndarray = np.empty((0, 4))

    def vis_boxes(self, thickness: int = 2) -> Image.Image:
        if self.boxes.shape[0] == 0:
            return self.image
        img = self.image.copy()
        drawer = Draw(img)

        for box in self.boxes:
            drawer.rectangle(
                box.tolist(),
                width=thickness,
                outline="white"
            )
        return img

    @property
    def normed_boxes(self) -> np.ndarray:
        return self.boxes / np.array([self.image.width, self.image.height, self.image.width, self.image.height])

    @property
    def boxes_crops(self) -> List[Image.Image]:
        return [
            self.image.crop(b)
            for b in self.boxes
        ]

    @property
    def normed_boxes_x_y_w_h(self) -> np.ndarray:
        boxes_x_y_w_h = self.normed_boxes
        boxes_x_y_w_h[:, 2:4] = boxes_x_y_w_h[:, 2:4] - boxes_x_y_w_h[:, 0:2]
        return boxes_x_y_w_h


@dc.dataclass(frozen=True)
class MaskedExample(BoxedExample):
    masks: Optional[List[Image.Image]] = dc.field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path):
        obj: MaskedExample = super().from_path(path)
        masks = []
        boxes = np.empty((0, 4))

        mask_path = path / _MASKS_FOLDER_NAME

        if mask_path.exists():
            masks_arrays = np.array([
                (imageio.imread(m) != 0).astype("uint8") for m in sorted(list(mask_path.iterdir()))
            ])
            masks = [Image.fromarray(m)  for m in masks_arrays]

            boxes = np.array([
                box_from_single_segmask(np.array(m))
                for m in masks
            ])
        return dc.replace(
            obj,
            masks=masks,
            boxes=boxes
        )

    @property
    def mask_array(self) -> np.ndarray:
        return np.array([np.array(mimg) for mimg in self.masks])

    def vis_segmasks(self, alpha: float = 0.5):
        masks = self.mask_array.sum(axis=0)

        plt.imshow(self.image)
        plt.imshow(masks, alpha=alpha)
        plt.show()
