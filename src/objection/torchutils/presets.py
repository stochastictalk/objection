import torch
from ..transforms import (
    Compose,
    RandomHorizontalFlip,
    PILToTensor,
    ConvertImageDtype,
    ScaleJitter,
    FixedSizeCrop,
    RandomPhotometricDistort,
    RandomZoomOut,
    RandomShortestSize,
    RandomIoUCrop
) 


class DetectionPresetTrain:
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        if data_augmentation == "hflip":
            self.transforms = Compose(
                [
                    RandomHorizontalFlip(p=hflip_prob),
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "lsj":
            self.transforms = Compose(
                [
                    ScaleJitter(target_size=(1024, 1024)),
                    FixedSizeCrop(size=(1024, 1024), fill=mean),
                    RandomHorizontalFlip(p=hflip_prob),
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "multiscale":
            self.transforms = Compose(
                [
                    RandomShortestSize(
                        min_size=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800), max_size=1333
                    ),
                    RandomHorizontalFlip(p=hflip_prob),
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssd":
            self.transforms = Compose(
                [
                    RandomPhotometricDistort(),
                    RandomZoomOut(fill=list(mean)),
                    RandomIoUCrop(),
                    RandomHorizontalFlip(p=hflip_prob),
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssdlite":
            self.transforms = Compose(
                [
                    RandomIoUCrop(),
                    RandomHorizontalFlip(p=hflip_prob),
                    PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = Compose(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)
