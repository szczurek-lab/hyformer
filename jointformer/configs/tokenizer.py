
from typing import List, Optional, Union, Callable
from jointformer.configs.base import Config


class TokenizerConfig(Config):

    def __init__(
            self,
            tokenizer: Optional[Union[Callable, List]],
            path_to_vocabulary: str,
            max_molecule_length: int,
            set_separate_task_tokens: bool
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.path_to_vocabulary = path_to_vocabulary
        self.max_molecule_length = max_molecule_length
        self.set_separate_task_tokens = set_separate_task_tokens
