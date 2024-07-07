from typing import Optional

from jointformer.configs.base import Config


class ModelConfig(Config):

    def __init__(
        self,
        model_name: str,
        embedding_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        bias: Optional[bool] = None,
        dropout: Optional[float] = None,
        layer_norm_eps: Optional[float] = None,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        prediction_task: Optional[str] = None,
        num_prediction_tasks: Optional[int] = None,
        num_physchem_tasks: Optional[int] = None,
        pretrained_filepath: Optional[int] = None,
        predictor_hidden_size: Optional[int] = None,
        predictor_dropout: Optional[float] = None
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
        self.prediction_task = prediction_task
        self.num_prediction_tasks = num_prediction_tasks
        self.num_physchem_tasks = num_physchem_tasks
        self.pretrained_filepath = pretrained_filepath
        self.predictor_hidden_size = predictor_hidden_size
        self.predictor_dropout = predictor_dropout
