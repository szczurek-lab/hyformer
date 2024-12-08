
from typing import List, Optional, Union, Callable
from jointformer.configs.base import BaseConfig


class DatasetConfig(BaseConfig):

    def __init__(
            self,
            dataset_name: Optional[str] = None,
            path_to_train_data: Optional[str] = None,
            path_to_val_data: Optional[str] = None,
            path_to_test_data: Optional[str] = None,
            transform: Optional[Union[Callable, List]] = None,
            target_transform: Optional[Union[Callable, List]] = None,
            task_type: Optional[str] = None,
            task_metric: Optional[str] = None,
            num_tasks: Optional[int] = None
    ):
        super().__init__()
        self.path_to_train_data = path_to_train_data
        self.path_to_val_data = path_to_val_data    
        self.path_to_test_data = path_to_test_data
        self.dataset_name = dataset_name
        self.transform = transform
        self.target_transform = target_transform
        self.task_type = task_type
        self.task_metric = task_metric
        self.num_tasks = num_tasks
