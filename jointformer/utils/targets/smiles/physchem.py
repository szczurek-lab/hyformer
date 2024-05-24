import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem.QED import qed
from jointformer.utils.targets.smiles.molbert.featurizer import PhysChemFeaturizer

from jointformer.utils.targets.smiles.base import BaseTarget


NORMALIZE = True


class PhysChem(BaseTarget):
    """ PhysChem target. """

    def __init__(self):
        super().__init__()
        self.descriptor_set = 'all'
        self.num_physchem = 200
        self.descriptor_list = PhysChemFeaturizer.get_descriptor_subset(self.descriptor_set, self.num_physchem)
        self.physchem_featurizer = PhysChemFeaturizer(descriptors=self.descriptor_list, normalise=True)

    def _get_target(self, example: str) -> float:
        physchem, valid = self.physchem_featurizer.transform_single(example)
        assert bool(valid), f'Cannot compute the physchem props for {example}'
        return physchem[:self.num_physchem].tolist()

    @property
    def target_names(self):
        return ["physchem"]

    def __repr__(self):
        return "PhysChem"

    def __str__(self):
        return "PhysChem"

    def __len__(self):
        return 200
