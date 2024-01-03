import torch.nn

from hybrid_transformer.configs.model import ModelConfig
from hybrid_transformer.models.pre_train import GPTPreTrained, BERTPreTrained, HybridTransformerPreTrained


PREDICTION_MODEL_CONFIGS = {
    'GPTForPrediction': './configs/models/prediction/gpt_finetune/config.json',
    'JointGPT': './configs/models/prediction/gpt_joint/config.json',
    # 'JointGPTNonLikelihood': './configs/models/prediction/gpt_joint/config.json',
    # 'HybridTransformerGPTInit': './configs/models/prediction/hybrid_transformer_gpt_init/config.json',
    'HybridTransformer': './configs/models/prediction/hybrid_transformer/config.json',
    'HybridTransformerWithPenalty': './configs/models/prediction/hybrid_transformer_penalty/config.json'
}

MLM_PREDICTION_MODELS = [model_key for model_key in PREDICTION_MODEL_CONFIGS.keys() if 'HybridTransformer' in model_key or 'BERT' in model_key]
LM_PREDICTION_MODELS = [model_key for model_key in PREDICTION_MODEL_CONFIGS.keys() if model_key not in MLM_PREDICTION_MODELS]


class GPTForPrediction(GPTPreTrained):
    """ A GPT-like model, with a prediction head, used for a prediction task. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        if outputs['supervised_loss'] is not None:
            outputs['loss'] = outputs['supervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'GPTForPrediction':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads)


class BERTForPrediction(BERTPreTrained):
    """ A BERT-like model, with a prediction head, used for a prediction task. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        if outputs['supervised_loss'] is not None:
            outputs['loss'] = outputs['supervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'BERTForPrediction':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads)


class JointGPT(GPTPreTrained):
    """ A GPT-like model, with a prediction head, used jointly for a language modeling and prediction task. """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        if outputs['supervised_loss'] is not None and outputs['unsupervised_loss'] is not None:
            outputs['loss'] = outputs['supervised_loss'] + outputs['unsupervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'JointGPT':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads)


class JointGPTNonLikelihood(GPTPreTrained):
    """ A GPT-like model, with a prediction head, used jointly for a language modeling and prediction task. """
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = torch.nn.Parameter(data=torch.Tensor([alpha]), requires_grad=False)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        if outputs['supervised_loss'] is not None and outputs['unsupervised_loss'] is not None:
            outputs['loss'] = self.alpha * outputs['supervised_loss'] + outputs['unsupervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'JointGPTNonLikelihood':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads,
            alpha=config.alpha)


class HybridTransformer(HybridTransformerPreTrained):
    """ Hybrid Transformer, used jointly for a language modeling and prediction task.

    Can be initialized both from a GPT-like model pre-trained using only a language modelling task or from a unified
    model pre-trained with a combination of language modelling and masked language modelling tasks. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        if kwargs['task'] == 'lm':
            if outputs['unsupervised_loss'] is not None:
                outputs['loss'] = outputs['unsupervised_loss']
        if kwargs['task'] == 'mlm':
            if outputs['supervised_loss'] is not None:
                outputs['loss'] = outputs['supervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'HybridTransformer':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads,
            task_p=config.task_p)


class HybridTransformerWithPenalty(HybridTransformerPreTrained):
    """ Hybrid Transformer, used jointly for a language modeling and prediction task.

    Can be initialized both from a GPT-like model pre-trained using only a language modelling task or from a unified
    model pre-trained with a combination of language modelling and masked language modelling tasks.

    Adds a pseudo-likelihood penalty to the loss function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        if kwargs['task'] == 'lm':
            if outputs['unsupervised_loss'] is not None:
                outputs['loss'] = outputs['unsupervised_loss']
        if kwargs['task'] == 'mlm':
            if outputs['supervised_loss'] is not None and outputs['unsupervised_loss'] is not None:
                outputs['loss'] = outputs['supervised_loss'] + outputs['unsupervised_loss']
        return outputs

    @classmethod
    def from_config(cls, config: ModelConfig) -> 'HybridTransformerWithPenalty':
        return cls(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            num_layers=config.num_layers,
            bias=config.bias,
            num_heads=config.num_heads,
            task_p=config.task_p)
