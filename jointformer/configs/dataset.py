
from typing import List, Optional, Union, Callable
from jointformer.configs.base import Config


class DatasetConfig(Config):

    def __init__(
            self,
            path_to_train_data: Optional[str] = None,
            path_to_train_properties: Optional[str] = None,
            path_to_val_data: Optional[str] = None,
            path_to_val_properties: Optional[str] = None,
            path_to_test_data: Optional[str] = None,
            path_to_test_properties: Optional[str] = None,
            dataset_name: Optional[str] = None,
            target_label: Optional[str] = None,
            data_filepath: Optional[str] = None,
            tatget_filepath: Optional[str] = None,
            validate: Optional[bool] = None,
            standardize: Optional[bool] = None,
            num_samples: Optional[int] = None,
            split: Optional[float] = None,
            splitter: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            task_type: Optional[str] = None,
            metric: Optional[str] = None,
            num_tasks: Optional[int] = None
    ):
        super().__init__()
        self.path_to_train_data = path_to_train_data
        self.path_to_train_properties = path_to_train_properties
        self.path_to_val_data = path_to_val_data    
        self.path_to_val_properties = path_to_val_properties
        self.path_to_test_data = path_to_test_data
        self.path_to_test_properties = path_to_test_properties
        self.dataset_name = dataset_name
        self.target_label = target_label
        self.data_filepath = data_filepath
        self.tatget_filepath = tatget_filepath
        self.validate = validate
        self.standardize = standardize
        self.num_samples = num_samples
        self.split = split
        self.splitter = splitter
        self.transform = transform
        self.target_transform = target_transform
        self.task_type = task_type
        self.metric = metric
        self.num_tasks = num_tasks
