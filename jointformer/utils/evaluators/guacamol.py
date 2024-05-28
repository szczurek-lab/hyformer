import os
import math
import warnings

from typing import Optional

from jointformer.utils.evaluators.base import Evaluate
from jointformer.utils.data import read_strings_from_file


class GuacamolEvaluator(Evaluate):

    def __init__(
            self,
            generated_file_path: str,
            reference_file_path: str,
            out_dir: Optional[str] = None,
            device: Optional[str] = None,
    ) -> None:
        super().__init__(generated_file_path, reference_file_path, out_dir, device)
        self.load_reference_data()

    def load_reference_data(self):
        self.reference_data = read_strings_from_file(self.reference_file_path)

    def evaluate(self):

        if len(self.generated_data) < 30000:
            warnings.warn("MOSES metrics require at least 30k samples")

        self.results = self.get_all_metrics(self.generated_data, self.reference_data, device=self.device)
        self.results['FCDGuacaMol/Test'] = math.exp(-0.2 * self.results['FCD/Test'])
        self.results['FCDGuacaMol/TestSF'] = math.exp(-0.2 * self.results['FCD/TestSF'])

    def _set_output_filename(self):
        self.results_filename = "results_guacamol.txt"
        if self.out_dir is not None:
            self.results_filename = os.path.join(self.out_dir, self.results_filename)

    @staticmethod
    def get_all_metrics(generated_data, reference_data, device) -> dict:
        out = {}

        generated_data_valid = None
        generated_data_unique = None
        generated_data_canonical = None

        if len(generated_data) < 10000 or len(generated_data_canonical) < 10000:
            warnings.warn("GuacaMol metrics require at least 10k samples")

        out['Validity'] = calculate_validity(generated_data, device)
        out['Uniqueness'] = calculate_uniqueness(generated_data_valid, device)
        out['Novelty'] = calculate_novelty(generated_data_canonical, reference_data, device)
        out['KlDiv'] = None
        out['FCD'] = None

        return out
