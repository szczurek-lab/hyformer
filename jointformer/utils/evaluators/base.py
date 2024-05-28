import os
import json
import torch

from typing import Optional

from jointformer.utils.data import read_strings_from_file, write_dict_to_file


class Evaluate:

    def __init__(
            self,
            generated_file_path: str,
            reference_file_path: Optional[str] = None,
            out_dir: Optional[str] = None,
            device: Optional[str] = None,
            seed: Optional[int] = 0
    ):

        self.generated_file_path = generated_file_path
        self.reference_file_path = reference_file_path
        self.out_dir = out_dir
        self.device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else device
        self.seed = seed

        self.generated_data = None
        self.reference_data = None
        self.results_file_name = None
        self.results_file_path = None
        self.results = {}

        self.load_generated_data()

    def save(self) -> None:
        if self.out_dir:
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
            write_dict_to_file(self.results, self.results_file_path)

    def load_generated_data(self):
        self.generated_data = read_strings_from_file(self.generated_file_path)

    def load_reference_data(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _set_output_filename(self):
        raise NotImplementedError
