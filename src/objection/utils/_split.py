from random import sample
from typing import Tuple, List

from objection._ObjectDetectionDataset import ObjectDetectionDataset

def split(
        data: ObjectDetectionDataset,
        training_fraction: float = 0.75,
        training_transforms: List = []
    ) -> Tuple[ObjectDetectionDataset, ObjectDetectionDataset]:
    """Randomly splits data into two disjoint ObjectDetectionDataset datasets,
    one of which is initialized with `transforms=training_transforms`.
    
    Parameters
    ----------
    data : ObjectDetectionDataset
        The dataset to derive the partitions from.
    
    training_fraction : float, default=0.75
        The proportion of records from data to include in the training partition.
    
    training_transforms : List
        List of augmentations to apply to the training data.
    
    Returns
    -------
    Tuple[ObjectDetectionDataset, ObjectDetectionDataset]
        Training and validation datasets.
    """
    
    all_ix = range(len(data))
    training_ix = sample(all_ix, int(training_fraction*len(data)))
    validation_ix = list(set(all_ix) - set(training_ix))
    
    training_data = ObjectDetectionDataset(
        data.annotations_filepath, 
        index_subset=training_ix,
        transforms=training_transforms
    )
    validation_data = ObjectDetectionDataset(
        data.annotations_filepath,
        index_subset=validation_ix,
        transforms=[]
    )
    return training_data, validation_data