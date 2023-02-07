from typing import Generator, List

import numpy as np
import torch


from .models._get_model import get_model
from .torchutils.engine import train_one_epoch, evaluate
from ._ObjectDetectionDataset import ObjectDetectionDataset


class ObjectDetectionModel:

    def __init__(
        self,
        n_classes: int = 2,
        mode: str = "fast"
        ):
        self.n_classes = n_classes
        self.mode = mode

    def fit(
        self,
        training_dataset: ObjectDetectionDataset,
        validation_dataset: ObjectDetectionDataset,
        n_epochs: int = 10
        ):
        """
        Parameters
        ----------
        training_dataset : ObjectDetectionDataset

        validation_dataset : ObjectDetectionDataset

        n_epochs : int

        Returns
        -------
        None
        """

        # Train on the GPU or on the CPU, if a GPU is not available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Get the model.
<<<<<<< HEAD
        self.model = get_model(self.n_classes)
        self.model.to(device)
=======
        model = get_model(self.n_classes)
        model.to(device)
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b

        # Get dataloaders.
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=2,
            shuffle=True,
<<<<<<< HEAD
            # num_workers=4,
            collate_fn=lambda batch: tuple(zip(*batch)) # merges a list of samples to form a mini-batch of Tensors
=======
            num_workers=4,
            collate_fn=lambda batch: tuple(zip(*batch))
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b
        )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
<<<<<<< HEAD
            # num_workers=4,
=======
            num_workers=4,
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        # Get optimizer.
        optimizer = torch.optim.Adam(
<<<<<<< HEAD
            [p for p in self.model.parameters() if p.requires_grad]
=======
            [p for p in model.parameters() if p.requires_grad]
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b
        )

        for epoch in range(n_epochs):
            train_one_epoch(
<<<<<<< HEAD
                self.model,
=======
                model,
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b
                optimizer,
                training_dataloader,
                device,
                epoch,
                print_freq=10
            )
            
            evaluate(
<<<<<<< HEAD
                self.model,
=======
                model,
>>>>>>> d5245f28ea16f66e7147fba2dec779ebe227cc5b
                validation_dataloader,
                device=device
            )
    
    def evaluate(
        dataset : ObjectDetectionDataset
    ):
        raise NotImplementedError()

    def write():
        raise NotImplementedError()