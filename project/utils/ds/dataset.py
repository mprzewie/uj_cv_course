from pathlib import Path
from typing import List, Callable, Dict

from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from project.utils.ds.structures import BoxedExample, MaskedExample
from project.utils.ds import transforms as tr
from project.utils.ds import random_transforms as rnd

transform_chain = [
    tr.to_grayscale,
    tr.flip_color_on_intensity_heuristic,
    tr.max_pixel_to_255,
    rnd.random_transform(tr.rotate, rnd.rotate_90_sampler()),
    rnd.random_transform(tr.rotate, rnd.rotate_sampler(amplitude=15)),
    rnd.random_transform(tr.resize, rnd.resize_sampler(amplitude=0.5)),
    rnd.random_transform(tr.crop, rnd.constant_crop_size_sampler(crop_size=0.7)),
    # tr.to_tensors_tuple
]


class BoxedExamplesDataset(Dataset):
    def __init__(self, examples: List[BoxedExample], transform: Callable = lambda x: x):
        """
        Args:
            examples:
            transform:
        """
        self.examples = examples
        self.transform = transform

    def __getitem__(self, item: int):
        return self.transform(self.examples[item])

    def __len__(self):
        return len(self.examples)

    @classmethod
    def from_path(cls, path: Path, **kwargs):
        return cls(
            [
                MaskedExample.from_path(p).replace(masks=None)
                for p in tqdm(path.iterdir())
            ],
            **kwargs
        )
