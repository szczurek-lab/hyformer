""" Base class for all molecular properties (targets). """

import numpy as np

from tqdm import tqdm
from typing import List, Union

_TARGET_DTYPE = np.float32


class BaseTarget:
    """
    Base class for calculating molecular properties (targets).

    Parameters
    ----------
    verbose : bool, optional
        If True, display progress information (default is True).
    """

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the BaseTarget with verbosity control.

        Args:
            verbose (bool): If True, display progress information.
        """
        self.verbose = verbose

    def __call__(self, examples: Union[List[str], str]) -> np.ndarray:
        """
        Calculate targets for given examples.

        Parameters
        ----------
        examples : list of str or str
            A list of example strings or a single example string.

        Returns
        -------
        np.ndarray
            An array of calculated targets.
        """
        if isinstance(examples, str):
            examples = [examples]

        targets = np.zeros((len(examples), len(self)), dtype=_TARGET_DTYPE)
        for idx, example in enumerate(tqdm(examples, desc="Calculating target data", disable=not self.verbose)):
            targets[idx, :] = self._calculate_target(example)
        return targets

    def __len__(self) -> int:
        """
        Return the length of the target vector. Must be implemented by subclasses.

        Returns
        -------
        int
            The length of the target vector.
        """
        pass

    def _calculate_target(self, example: str) -> Union[float, np.ndarray, List[float]]:
        """
        Calculate the target for a single example. Must be implemented by subclasses.

        Parameters
        ----------
        example : str
            A single example string.

        Returns
        -------
        float or np.ndarray
            The calculated target.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @property
    def target_names(self) -> List[str]:
        """
        Get the names of the targets. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this property.")
    