import math
from typing import Generator, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .models._get_model import get_model
from .torchutils.engine import evaluate
from ._ObjectDetectionDataset import ObjectDetectionDataset
from ._TrainingConfiguration import TrainingConfiguration

class ObjectDetectionModel:

    def __init__(
        self,
        n_classes: int = 2,
        mode: str = "fast"
        ):
        self.n_classes = n_classes
        self.mode = mode
        self.writer = SummaryWriter()
        self.tag_loss_mapping = {
            "Classifier loss/": "loss_classifier",
            "Bounding box regression loss/": "loss_box_reg",
            "Segmentation mask loss/": "loss_mask",
            "Objectness loss/": "loss_objectness",
            "RPN bounding box regression loss/": "loss_rpn_box_reg"
        }

    def fit(
        self,
        training_dataset: ObjectDetectionDataset,
        validation_dataset: ObjectDetectionDataset,
        configuration: TrainingConfiguration = None
        ):
        """
        Parameters
        ----------
        training_dataset : ObjectDetectionDataset

        validation_dataset : ObjectDetectionDataset

        configuration : TrainingConfiguration

        Returns
        -------
        None
        """

        # Train on the GPU or on the CPU, if a GPU is not available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Get the model and set it to training mode.
        self.model = get_model(self.n_classes)
        self.model.to(device)

        # Get dataloaders.
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=2,
            shuffle=True,
            # num_workers=4,
            collate_fn=lambda batch: tuple(zip(*batch)) # merges a list of samples to form a mini-batch of Tensors
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            # num_workers=4,
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        # Get optimizer.
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad]
        )

        # Fit the model.
        n_epochs = 100
        global_step = 0
        for epoch_j in range(n_epochs):

            # Minibatch gradient descent.
            self.model.train()
            for batch_j, (images, targets) in enumerate(training_dataloader):
                # Increment global step.
                global_step += 1

                # Load images and targets onto GPU.
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Make forward pass.
                loss_dict = self.model(images, targets)
                for loss_tag, loss_key in self.tag_loss_mapping.items():
                    self.writer.add_scalar(f"{loss_tag}train", loss_dict[loss_key], global_step)

                # Compute total loss.
                total_loss = sum(loss for loss in loss_dict.values())
                loss_value = total_loss.item()
                self.writer.add_scalar("Total loss/train", total_loss, global_step)

                # Check loss is not NaN.
                if not math.isfinite(loss_value):
                    break

                # Compute gradients and update parameters.
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Evaluation.
            from .utils._HiddenPrints import HiddenPrints # To avoid a circular dependency.
            with HiddenPrints():
                coco_evaluator = evaluate(self.model, validation_dataloader, device) # Type CocoEvaluator.
            print(coco_evaluator)
            #for loss_tag, loss_key in self.tag_loss_mapping.items():
            #    self.writer.add_scalar(f"{loss_tag}validation", loss_dict[loss_key], global_step)
            #self.writer.add_scalar("Total loss/validation", total_loss, global_step)

        self.writer.flush()

    
    def evaluate(
        dataset : ObjectDetectionDataset
    ):
        raise NotImplementedError()

    def write():
        raise NotImplementedError()
