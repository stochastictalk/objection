from pathlib import Path

import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

WEIGHT_PATH = Path(__file__).parent.parent / "_models" / "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"

def get_model(num_classes: int) \
    -> torchvision.models.detection.mask_rcnn.MaskRCNN:
    """
    num_classes includes background class e.g. PennFudan has
    two classes - background and person.
    
    model.__call__ 
    * List[FloatTensor] -> returns List[Dict] where each Dict has keys "boxes", "labels", "scores".
    * FloatTensors can be different sizes, must be three channels.
    * Gradients are attached to "boxes", "labels", "scores" tensors.
    
    """
    # Load an instance segmentation model pre-trained on COCO.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights_backbone=None)
    model.load_state_dict(torch.load(WEIGHT_PATH))


    # Replace the pre-trained box predictor head with a new one.
    in_features = model.roi_heads.box_predictor.cls_score.in_features    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the pre-trained mask predictor head with a new one.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels = in_features_mask, # Number of input channels.
        dim_reduced = 256, # Number of hidden layer units.
        num_classes = num_classes # Number of object classes.
    )

    return model