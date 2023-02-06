from typing import List

import numpy as np

def get_bbox_from_mask(mask: np.ndarray) -> List[int]:
    # boolean mask has shape (height, width)
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return [xmin, ymin, xmax, ymax]