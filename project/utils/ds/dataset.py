from pathlib import Path
from typing import List, Callable, Dict

from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from project.utils.ds.structures import BoxedExample, MaskedExample


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