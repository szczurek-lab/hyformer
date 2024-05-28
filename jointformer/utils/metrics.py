import logging

from typing import List, Optional

from jointformer.utils.chemistry import is_valid, canonicalize_list

logger = logging.getLogger(__name__)

REQUIRED_GUACAMOL_DATA_LENGTH = 10000


class Validity:
    """ Calculate validity as a fraction of valid SMILES strings in the list of SMILES strings. """

    def __init__(self, num_samples: Optional[int] = 10000):
        self.num_samples = num_samples

    def __call__(self, smiles_list: List[str]):
        return self._calculate_validity(smiles_list)

    def _calculate_validity(self, smiles_list: List[str]):
        smiles_list = self.subset(smiles_list)
        if len(smiles_list) < REQUIRED_GUACAMOL_DATA_LENGTH:
            logger.warning(f"Dataset is too small for reliable evaluation of validity: {len(smiles_list)} < {REQUIRED_GUACAMOL_DATA_LENGTH}")
        valid_smiles = [smiles for smiles in smiles_list if is_valid(smiles)]
        return len(valid_smiles) / len(smiles_list)

    def subset(self, smiles: List[str]):
        return smiles[:self.num_samples] if len(smiles) > self.num_samples else smiles


class Uniqueness:
    """ Calculate uniqueness as a fraction of unique SMILES strings in the list of valid SMILES strings. """

    def __init__(self, num_samples: Optional[int] = 10000):
        self.num_samples = num_samples

    def __call__(self, smiles_list: List[str]):
        return self._calculate_uniqueness(smiles_list)

    def _calculate_uniqueness(self, smiles_list: List[str]):
        smiles_list = self.subset(smiles_list)
        if len(smiles_list) < REQUIRED_GUACAMOL_DATA_LENGTH:
            logger.warning(
                f"Dataset is too small for reliable evaluation of uniqueness: {len(smiles_list)} < {REQUIRED_GUACAMOL_DATA_LENGTH}")
        unique_smiles_list = canonicalize_list(smiles_list, include_stereocenters=False)
        return len(unique_smiles_list) / len(smiles_list)

    def subset(self, smiles_list: List[str]):
        smiles_list = [smiles for smiles in smiles_list if is_valid(smiles)]
        return smiles_list[:self.num_samples] if len(smiles_list) > self.num_samples else smiles_list


class Novelty:
    """ Calculate novelty as a fraction of SMILES strings not present in the training dataset, after canonicalization.
     """
    def __init__(self, num_samples: Optional[int] = 10000, train_smiles: List[str] = None):
        self.num_samples = num_samples
        self.train_smiles = set(canonicalize_list(train_smiles, include_stereocenters=False))

    def __call__(self, smiles_list: List[str]):
        return self._calculate_novelty(smiles_list)

    def _calculate_novelty(self, smiles_list: List[str]):
        smiles_list = self.subset(smiles_list)
        if len(smiles_list) < REQUIRED_GUACAMOL_DATA_LENGTH:
            logger.warning(
                f"Dataset is too small for reliable evaluation of novelty: {len(smiles_list)} < {REQUIRED_GUACAMOL_DATA_LENGTH}")
        novel_molecules = smiles_list.difference(self.training_set_molecules)

        novel_ratio = len(novel_molecules) / self.number_samples
        return len(set(unique_smiles) - set(self.train_smiles)) / len(unique_smiles)

    def subset(self, smiles_list: List[str]):
        smiles_list = canonicalize_list(smiles_list, include_stereocenters=False)
        return smiles_list[:self.num_samples] if len(smiles_list) > self.num_samples else smiles_list


class PhysChemKlDiv:

    def __init__(self, num_samples: Optional[int] = 10000):
        self.num_samples = num_samples