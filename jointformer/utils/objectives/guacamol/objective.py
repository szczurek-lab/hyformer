"""Get objective values. """

import torch

from typing import List

from jointformer.utils.objectives.guacamol.utils import smiles_to_desired_scores

DTYPE_OBJECTIVE = torch.float32

GUACAMOL_TASKS = [
    'zaleplon',
    'osimertinib',
    'fexofenadine',
    'ranolazine',
    'perindopril',
    'amlodipine',
    'sitagliptin'
]


def get_objective(smiles_list: List[str], objective: str, verbose=False) -> torch.Tensor:
    return torch.Tensor(smiles_to_desired_scores(smiles_list, objective, verbose)).to(DTYPE_OBJECTIVE).unsqueeze(1)
