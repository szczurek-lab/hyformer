import torch

from abc import ABC, abstractmethod
from typing import List


class AutoTokenizerBase:

    @classmethod
    def from_config(cls, config) -> "AutoTokenizerBase":
        return None
