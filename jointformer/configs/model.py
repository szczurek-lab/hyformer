from typing import Optional

from jointformer.configs.base import Config


class ModelConfig(Config):

    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        layer_norm_eps: float,
        vocab_size: int,
        max_seq_len: int,
        task_p: Optional[float] = None,
        prediction_task: Optional[str] = None,
        num_prediction_tasks: Optional[int] = None,
        num_physchem_tasks: Optional[int] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.task_p = task_p
        self.prediction_task = prediction_task
        self.num_prediction_tasks = num_prediction_tasks
        self.num_physchem_tasks = num_physchem_tasks
