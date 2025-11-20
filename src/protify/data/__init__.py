from .supported_datasets import (
    supported_datasets,
    internal_datasets,
    possible_with_vector_reps,
    standard_data_benchmark,
    testing,
)

try:
    from .dataset_descriptions import dataset_descriptions
except ImportError:
    dataset_descriptions = {}

from .dataset_utils import list_supported_datasets, get_dataset_info 