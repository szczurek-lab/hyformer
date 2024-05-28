import os
import math
import warnings

from typing import Optional

from moses.metrics.metrics import get_all_metrics

from jointformer.utils.evaluators.base import Evaluate


class MosesEvaluator(Evaluate):

    def __init__(
            self,
            generated_file_path: str,
            # reference_file_path: Optional[str] = None, # for now, use the MOSES default reference data
            out_dir: Optional[str] = None,
            device: Optional[str] = None
    ) -> None:
        super().__init__(generated_file_path, None, out_dir, device)
        self._set_output_file_path()

    def evaluate(self):

        if len(self.generated_data) < 30000:
            warnings.warn("MOSES metrics require at least 30k samples")

        self.results = get_all_metrics(self.generated_data, device=self.device)
        self.results['FCDGuacaMol/Test'] = math.exp(-0.2 * self.results['FCD/Test'])
        self.results['FCDGuacaMol/TestSF'] = math.exp(-0.2 * self.results['FCD/TestSF'])

    def _set_output_file_path(self):
        self.results_file_name = "results_moses.txt"
        if self.out_dir is not None:
            self.results_file_path = os.path.join(self.out_dir, self.results_file_name)
