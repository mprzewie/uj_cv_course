import numpy as np


def box_from_single_segmask(segmask: np.ndarray) -> np.ndarray:
    ones_inds = np.argwhere(segmask==1)
    return np.array([
        np.min(ones_inds[:, 1]),
        np.min(ones_inds[:, 0]),
        np.max(ones_inds[:, 1]),
        np.max(ones_inds[:, 0])
    ])