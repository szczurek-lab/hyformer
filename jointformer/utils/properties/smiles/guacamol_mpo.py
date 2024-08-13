import numpy as np

from guacamol import standard_benchmarks
from typing import Optional

from jointformer.utils.properties.smiles.base import BaseTarget

GUACAMOL_MPO_DEFAULT_DTYPE = np.float64

GUACAMOL_MPO_TASK_FN = {
    'amlodipine': standard_benchmarks.amlodipine_rings(),
    'fexofenadine': standard_benchmarks.hard_fexofenadine(),
    'osimertinib': standard_benchmarks.hard_osimertinib(),
    'perindopril': standard_benchmarks.perindopril_rings(),
    'sitagliptin': standard_benchmarks.sitagliptin_replacement(),
    'ranolazine': standard_benchmarks.ranolazine_mpo(),
    'zaleplon': standard_benchmarks.zaleplon_with_other_formula(),
}


class GuacamolMPO(BaseTarget):
    """ Guacamol MPO targets.
    """
    
    def _get_target(self, example: str, dtype: Optional[np.dtype] = GUACAMOL_MPO_DEFAULT_DTYPE) -> np.ndarray:
        try: 
            target = np.array(list(map(lambda fn: fn.objective.score(example), GUACAMOL_MPO_TASK_FN.values())), dtype=dtype)
            target[target < 0] = np.nan
            return target
        except Exception:
            return np.full(len(GUACAMOL_MPO_TASK_FN), np.nan)

    @property
    def target_names(self):
        return list(GUACAMOL_MPO_TASK_FN.keys())

    def __repr__(self):
        return "GuacamolMPO"

    def __str__(self):
        return "GuacamolMPO"
    
    def __len__(self):
        return len(GUACAMOL_MPO_TASK_FN)
