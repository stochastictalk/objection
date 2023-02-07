import pathlib
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from .utils import get_bbox_from_mask
<<<<<<< HEAD
from .torchutils.engine import train_one_epoch, evaluate
from .torchutils.utils import collate_fn
=======
from torchutils.engine import train_one_epoch, evaluate
from torchutils.utils import collate_fn
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b
from .torchutils import transforms as T

class ObjectDetectionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_paths=List[pathlib.Path],
        mask_paths=List[pathlib.Path],
        transforms=None
        ):
        """
        Parameters
        ----------
        transforms : List[@TODO]
            List of fortuna.torchutils.transforms classes.

        image_paths : List[pathlib.Path]
            List of image paths.

        mask_paths : List[pathlib.Path]
            List of image mask paths. What is the format of the masks?
        """
        self.default_transforms = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float)
            ])

        self.user_transforms = T.Compose(transforms)
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the example to retrieve.

        Returns
        -------
        Tensor, Dictionary
            Image and target.
        """
        # Get image and mask paths.
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        # Load image.
        image = Image.open(image_path).convert("RGB")
        
        # Load mask. NB object_ids denote distinct objects, not classes.
        mask = Image.open(mask_path)
        mask = np.array(mask)
        object_ids = np.unique(mask) # Get object ids from distinct values in mask.
        object_ids = object_ids[1:] # First id (0) is the background, so remove it.
        masks = (mask == object_ids[:, None, None]) # Each channel of masks corresponds to a different object id.

        # Extract bounding box coordinates from each segmentation mask.
        n_objects = len(object_ids)
        bboxes = [get_bbox_from_mask(masks[i, :, :]) for i in range(n_objects)]
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # Package into dict.
        target = {
            "boxes": bboxes, # Bounding boxes.
            "labels": torch.ones((n_objects,), dtype=torch.int64), # Class labels (only one class for PennFudan).
            "masks": torch.as_tensor(masks, dtype=torch.uint8), # Segmentation masks.
            "image_id": torch.tensor([index]),
            "area": (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]),
            "is_crowd": torch.zeros((n_objects,), dtype=torch.int64) # Set iscrowd = False for all.
        }

        # Apply default transforms.
        image, target = self.default_transforms(image, target)
        
        # Apply user-passed transforms.
        image, target = self.user_transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_paths)
    