import pathlib
from typing import List, Tuple
from PIL import Image

import numpy as np
import torch
from pycocotools.coco import COCO

from .utils import get_bbox_from_mask
from .torchutils.engine import train_one_epoch, evaluate
from .torchutils.utils import collate_fn
from .torchutils import transforms as T

class ObjectDetectionDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        annotations_filepath: pathlib.Path,
        transforms=[]
        ):
        """
        Parameters
        ----------
        annotations_filepath : pathlib.Path
            Filepath of image annotations in COCO format.

        transforms : List[@TODO], default=[]
            List of fortuna.torchutils.transforms classes.
        """
        self.annotations_filepath = pathlib.Path(annotations_filepath)
        self.default_transforms = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float)
            ])
        self.user_transforms = T.Compose(transforms)
        self.coco = COCO(annotations_filepath)

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
        filename = pathlib.Path(self.coco.imgs[index]["file_name"]).name
        image_path = self.annotations_filepath.parent / "images" / filename
        
        # Load image.
        image = Image.open(image_path).convert("RGB")
        
        # Load mask. NB object_ids denote distinct objects, not classes.
        # Get annotation ids for image.
        annotation_masks = []
        annotation_classes = []
        for annotation_id in self.coco.getAnnIds(index):
            annotation = self.coco.anns[annotation_id]
            annotation_masks.append(self.coco.annToMask(annotation))
            annotation_classes.append(annotation["category_id"])
        masks = np.stack(annotation_masks, axis=0) # Each channel is a different binary mask.

        # Cast each one to a mask and retrieve its class.
        #self.coco.annToMask() # Each channel of masks corresponds to a different object id.

        # Extract bounding box coordinates from each segmentation mask.
        n_objects = masks.shape[0]
        bboxes = [get_bbox_from_mask(masks[i, :, :]) for i in range(n_objects)]
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        # Package into dict.
        target = {
            "boxes": bboxes, # Bounding boxes.
            "labels": torch.as_tensor(annotation_classes, dtype=torch.int64), # Class labels (only one class for PennFudan).
            "masks": torch.as_tensor(masks, dtype=torch.uint8), # Segmentation masks.
            "image_id": torch.tensor([index]),
            "area": (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]),
            "is_crowd": torch.zeros((n_objects,), dtype=torch.int64), # Set iscrowd = False for all.
            "filename": filename
        }

        # Apply default transforms.
        image, target = self.default_transforms(image, target)
        
        # Apply user-passed transforms.
        image, target = self.user_transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_paths)