from hybrid_transformer.models.gpt import GPT
from hybrid_transformer.models.hybrid_transformer import HybridTransformer


class GPT(HybridTransformer):
    """ A GPT-like model, used for language modeling.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids, task='lm')


class BERT(HybridTransformer):
    """ A BERT-like model, used for masked language modeling.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids, task='mlm')


class GPTForRegressionTask(GPT):
    """ A GPT-like model, with a prediction head, used for a regression task. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BERTForRegressionTask('BERT'):
    """ A BERT-like model, with a prediction head, used for a regression task. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class JointGPTForRegressionTask(GPT):
    """ A GPT-like model, with a prediction head, used jointly for a language modeling and regression task. """
    def __init__(self, alpha: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha


class HybridTransformerForRegressionTask(HybridTransformer):
    """ Hybrid Transformer, used jointly for a language modeling and regression task.

    Can be initialized both from a GPT-like model pre-trained using only a language modelling task or from a unified
    model pre-trained with a combination of language modelling and masked language modelling tasks. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HybridTransformerWithPenaltyForRegressionTask(HybridTransformer):
    """ Hybrid Transformer, used jointly for a language modeling and regression task.

    Can be initialized both from a GPT-like model pre-trained using only a language modelling task or from a unified
    model pre-trained with a combination of language modelling and masked language modelling tasks.

    Adds a pseudo-likelihood penalty to the loss function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_ids):
        outputs = super().forward(input_ids=input_ids, task='mlm')
        outputs['loss'] = outputs['supervised_loss'] + outputs['unsupervised_loss']