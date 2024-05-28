import os
import math
import warnings

from typing import Optional

from jointformer.utils.metrics import calculate_validity, calculate_uniqueness, calculate_novelty, calculate_kl_div, calculate_fcd
from jointformer.utils.evaluators.base import Evaluate
from jointformer.utils.data import read_strings_from_file, get_random_subset
from jointformer.utils.chemistry import is_valid, canonicalize_list

NUM_SAMPLES = 10000


class GuacamolEvaluator(Evaluate):

    def __init__(
            self,
            generated_file_path: str,
            reference_file_path: str,
            out_dir: Optional[str] = None,
            device: Optional[str] = None,
            seed: Optional[int] = 0
    ) -> None:
        super().__init__(generated_file_path, reference_file_path, out_dir, device, seed)
        self.load_reference_data()
        self._set_output_filename()

    def load_reference_data(self):
        self.reference_data = read_strings_from_file(self.reference_file_path)
        self.reference_data = canonicalize_list(
            get_random_subset(self.reference_data, NUM_SAMPLES, seed=self.seed),
            include_stereocenters=False)

    def evaluate(self):

        if len(self.generated_data) < NUM_SAMPLES:
            warnings.warn("GuacaMol metrics require at least 10k samples")

        self.results = self.get_all_metrics(self.generated_data, self.reference_data)

    def _set_output_filename(self):
        self.results_file_name = "results_guacamol.txt"
        if self.out_dir is not None:
            self.results_file_path = os.path.join(self.out_dir, self.results_file_name)

    def get_all_metrics(self, generated_data, reference_data) -> dict:
        out = {}

        generated_data_valid = [smiles for smiles in generated_data if is_valid(smiles)]
        generated_data_canonical = canonicalize_list(generated_data_valid, include_stereocenters=False)

        if len(generated_data) < NUM_SAMPLES or len(generated_data_valid) < NUM_SAMPLES or len(generated_data_canonical) < NUM_SAMPLES:
            warnings.warn("GuacaMol metrics require at least 10k samples")

        out['Validity'] = calculate_validity(generated_data)
        out['Uniqueness'] = calculate_uniqueness(generated_data_valid)
        out['Novelty'] = calculate_novelty(generated_data_canonical, reference_data)
        out['KlDiv'] = calculate_kl_div(generated_data_canonical, reference_data)
        out['FCD'] = calculate_fcd(
            get_random_subset(generated_data, NUM_SAMPLES, seed=self.seed),
            reference_data
        )

        return out
