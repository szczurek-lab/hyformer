from typing import Optional

from jointformer.configs.base import Config
from jointformer.utils.runtime import find_multiple

EMBEDDING_DIM_HIDDEN_FACTOR = 8 / 3
EMBEDDIND_DIM_MULTIPLE_OF = 256


class ModelConfig(Config):

    def __init__(
        self,
        model_name: str,
        embedding_dim: Optional[int] = None,
        embedding_hidden_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_local_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        bias: Optional[bool] = None,
        attention_dropout: Optional[float] = None,
        feed_forward_dropout: Optional[float] = None,
        prediction_dropout: Optional[float] = None,
        layer_norm_eps: Optional[float] = None,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        prediction_task_type: Optional[str] = None,
        num_prediction_tasks: Optional[int] = None,
        num_physchem_tasks: Optional[int] = None,
        pretrained_filepath: Optional[int] = None,
        predictor_hidden_size: Optional[int] = None,
        predictor_dropout: Optional[float] = None,
        predictor_num_heads: Optional[int] = None,
        prediction_hidden_dim: Optional[int] = None,
        set_separate_task_tokens: Optional[bool] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.embedding_hidden_dim = embedding_hidden_dim
        self.num_heads = num_heads
        self.num_local_heads = num_local_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.bias = bias
        self.attention_dropout = attention_dropout
        self.feed_forward_dropout = feed_forward_dropout
        self.prediction_dropout = prediction_dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.prediction_task_type = prediction_task_type
        self.num_prediction_tasks = num_prediction_tasks
        self.num_physchem_tasks = num_physchem_tasks
        self.pretrained_filepath = pretrained_filepath
        self.predictor_hidden_size = predictor_hidden_size
        self.predictor_dropout = predictor_dropout
        self.predictor_num_heads = predictor_num_heads
        self.prediction_hidden_dim = prediction_hidden_dim
        self.set_separate_task_tokens = set_separate_task_tokens
        self._post_init()

    def _post_init(self):
        assert self.embedding_dim % self.num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        if self.embedding_hidden_dim is None:
            self.embedding_hidden_dim = find_multiple(self.embedding_dim * EMBEDDING_DIM_HIDDEN_FACTOR, EMBEDDIND_DIM_MULTIPLE_OF)
        if self.prediction_hidden_dim is None:
            self.prediction_hidden_dim = self.embedding_dim
        if self.num_local_heads is None:
            self.num_local_heads = self.num_heads
        if self.head_dim is None:
            self.head_dim = self.embedding_dim // self.num_heads
        if self.max_seq_len is not None:
            self.max_seq_len = find_multiple(self.max_seq_len, 8)
            