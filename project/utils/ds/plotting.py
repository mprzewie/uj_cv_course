from typing import Optional

import numpy as np


def random_colors(n_colors: int, seed: Optional[float] = 0) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.rand(n_colors, 3)

