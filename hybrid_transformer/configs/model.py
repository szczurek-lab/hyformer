from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):

    def __init__(
        self,
        model_name: str = 'GPT',
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        bias: bool = False,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        block_size: int = 1024,
        vocab_size: int = 588,
        max_seq_len: int = 128,
        **kwargs,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        super().__init__(**kwargs)

    def save(self, save_directory: str) -> None:
        super().save_pretrained(save_directory=save_directory)

    def load(self, config_path: str) -> None:
        super().from_pretrained(pretrained_model_name_or_path=config_path)
