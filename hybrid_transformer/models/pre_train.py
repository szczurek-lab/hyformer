from hybrid_transformer.models.base import Transformer
from hybrid_transformer.configs.model import ModelConfig


class HybridTransformerPreTrained(Transformer):
    """ Hybrid Transformer, used for both language modeling and masked language modeling.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs['loss'] = outputs['unsupervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'HybridTransformerPreTrained':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads,
            task_p=config.task_p)


class GPTPreTrained(HybridTransformerPreTrained):
    """ A GPT-like model, used for masked language modeling.
    """

    def __init__(self, **kwargs) -> None:
        kwargs['task_p'] = 1.0
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        kwargs['task'] = 'lm'
        return super().forward(**kwargs)

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'GPTPreTrained':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads)


class BERTPreTrained(HybridTransformerPreTrained):
    """ A BERT-like model, used for masked language modeling.
    """

    def __init__(self, **kwargs) -> None:
        kwargs['task_p'] = 0.0
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        kwargs['task'] = 'mlm'
        return super().forward(**kwargs)

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'BERTPreTrained':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads)
