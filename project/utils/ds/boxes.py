import numpy as np


def box_from_single_segmask(segmask: np.ndarray) -> np.ndarray:
    # print(segmask)
    ones_inds = np.argwhere(segmask == 1)
    if ones_inds.shape[0] == 0:
        print("shit")
        return np.array([0,0,0,0])
    return np.array([
        np.min(ones_inds[:, 1]),
        np.min(ones_inds[:, 0]),
        np.max(ones_inds[:, 1]),
        np.max(ones_inds[:, 0])
    ])


def areas(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def aspect_ratios(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1])


def centers(boxes: np.ndarray) -> np.ndarray:
    return np.array([
        boxes[:, 2] + boxes[:, 0],
        boxes[:, 3] + boxes[:, 1]
    ]).reshape(-1, 2) / 2
