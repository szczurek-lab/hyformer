"""Get objective values. """

import torch

from typing import List

from hybrid_transformer.utils.objectives.guacamol.utils import smiles_to_desired_scores

DTYPE_OBJECTIVE = torch.float32


def get_objective(smiles_list: List[str], objective: str, verbose=False) -> torch.Tensor:
    return torch.Tensor(smiles_to_desired_scores(smiles_list, objective, verbose)).to(DTYPE_OBJECTIVE).unsqueeze(1)
