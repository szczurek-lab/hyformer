from types import NoneType
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from hyformer.models.encoder import EncoderWrapper
from hyformer.models.llama_backbone import LLAMABackbone
from hyformer.models.utils import ModelOutput

from hyformer.utils.tokenizers.base import IGNORE_TOKEN_IDX
from hyformer.models.layers.prediction import PredictionHead
from hyformer.configs.model import ModelConfig
from hyformer.utils.decorators import inference

NAN_TARGET_IDX = -1


class Hyformer(LLAMABackbone):
    """ A joint transformer-based model for molecule generation and property prediction. Based on Llama backbone.
    """
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_embedding_dim: int, attention_dropout_p: float,
        num_transformer_layers: int, num_attention_heads: int, layer_norm_eps: float, num_prediction_tasks: int = None,
        prediction_prediction_task_type: str = None, prediction_head_dropout_p: float = None, init_weights: bool = True 
    ) -> None:
        """
        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary.
        embedding_dim : int
            The dimension of the embedding.
        hidden_embedding_dim : int
            The dimension of the hidden embedding.
        attention_dropout_p : float
            The dropout rate for the attention.
        num_transformer_layers : int
            The number of transformer layers.
        num_attention_heads : int
            The number of attention heads.
        layer_norm_eps : float
            The epsilon for the layer normalization.
        num_prediction_tasks : int, optional
            The number of prediction tasks, by default None.
        prediction_prediction_task_type : str, optional
            The type of prediction task, by default None.
        prediction_head_dropout_p : float, optional
            The dropout rate for the prediction head, by default None.
        init_weights : bool, optional
            Whether to initialize the model weights, by default True.
        """

        super().__init__(
            vocab_size=vocab_size, embedding_dim=embedding_dim,
            hidden_embedding_dim=hidden_embedding_dim, attention_dropout_p=attention_dropout_p, num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads, layer_norm_eps=layer_norm_eps, init_weights=False
        )
        
        # Initialize task specific heads
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        self.mlm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        
        # Weight tying https://paperswithcode.com/method/weight-tying
        self.token_embedding.weight = self.lm_head.weight
        self.mlm_head.weight = self.lm_head.weight
        
        # Weight initialization
        if init_weights:
            self.initialize_parameters()
            
        # Initialize prediction head using prediction specific init
        self.prediction_head = None
        self.prediction_prediction_task_type = prediction_prediction_task_type
        self.num_prediction_tasks = num_prediction_tasks
        if num_prediction_tasks is not None and prediction_prediction_task_type is not None:
            self.init_prediction_head(num_prediction_tasks=num_prediction_tasks, prediction_task_type=prediction_prediction_task_type, dropout_p=prediction_head_dropout_p)

    def init_prediction_head(self, num_prediction_tasks: int, prediction_task_type: str, dropout_p: float = None):
        self.prediction_head = PredictionHead(
            embedding_dim=self.embedding_dim,
            num_prediction_tasks=num_prediction_tasks,
            activation_fn='tanh' if prediction_task_type == 'classification' else 'relu',
            dropout_p=dropout_p
        )

    def load_pretrained(self, filename = None, state_dict = None, device='cpu', discard_prediction_head: bool = False):
        assert filename is not None or state_dict is not None, "Either filename or state_dict must be provided"
        assert filename is None or state_dict is None, "Only one of filename or state_dict must be provided"
        if filename is not None:
            state_dict = torch.load(filename, map_location=device, weights_only=True)['model']

        if discard_prediction_head:
            for k in list(state_dict.keys()):
                if 'prediction_head' in k:
                    state_dict.pop(k)
        
        super().load_state_dict(state_dict=state_dict)
    
    def resize_token_embeddings(self, new_vocab_size: int):
        
        old_vocab_size = self.vocab_size
        super().resize_token_embeddings(new_vocab_size)
        
        # Resize lm_head and mlm_head
        new_lm_head = nn.Linear(self.embedding_dim, new_vocab_size, bias=False)
        new_mlm_head = nn.Linear(self.embedding_dim, new_vocab_size, bias=False)
        
        new_lm_head.to(self.lm_head.weight.device)
        new_mlm_head.to(self.mlm_head.weight.device)
        
        with torch.no_grad():
            new_lm_head.weight[:old_vocab_size] = self.lm_head.weight
            new_mlm_head.weight[:old_vocab_size] = self.lm_head.weight
            
        self.lm_head = new_lm_head
        self.mlm_head = new_mlm_head
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        self.mlm_head.weight = self.token_embedding.weight

    def forward(
            self,
            input_ids: torch.Tensor,
            task: str,
            attention_mask: Optional[torch.Tensor] = None,
            next_token_only: Optional[bool] = False,
            past_key_values: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = False,
            input_labels: Optional[torch.Tensor] = None,
            target: Optional[torch.Tensor] = None,
            return_loss: bool = True,
            loss_fn_reduction: str = 'mean',
            nan_target_idx: int = -1,
            **kwargs
            ):
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            task: Task type ('lm', 'mlm', or 'prediction')
            attention_mask: Attention mask for the input
            next_token_only: Whether to return only the next token
            past_key_values: Past key values for caching
            use_cache: Whether to use caching
            input_labels: Labels for language modeling tasks
            target: Target values for prediction tasks
            return_loss: Whether to calculate and return loss
            loss_fn_reduction: Reduction method for loss calculation
            nan_target_idx: Index for NaN targets in prediction tasks
            **kwargs: Additional arguments
            
        Returns:
            ModelOutput containing model outputs and optionally loss
        """
        if task == 'lm':
            _is_causal = True
            _attention_mask = None
        else:
            _is_causal = False
            _attention_mask = attention_mask
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=_attention_mask,
            is_causal=_is_causal,
            next_token_only=next_token_only,
            past_key_values=past_key_values,
            use_cache=use_cache
            )

        logits = None
        loss = None
        
        # Calculate task-specific outputs
        if task == 'lm':
            logits = self.lm_head(outputs['embeddings'])
        elif task == 'mlm':
            logits = self.mlm_head(outputs['embeddings'])
        elif task == 'prediction':
            _cls_token_idx = 0
            logits = self.prediction_head(outputs['embeddings'][:, _cls_token_idx])
        else:
            raise ValueError(f'Variable `task` must be either `lm`, `mlm`, or `prediction` and {task} was given.')

        # Calculate loss if requested
        if return_loss:
            loss = self._calculate_loss(
                logits=logits,
                task=task,
                input_labels=input_labels,
                target=target,
                loss_fn_reduction=loss_fn_reduction,
                nan_target_idx=nan_target_idx
            )

        return ModelOutput(
            embeddings=outputs['embeddings'],
            logits=logits,
            attention_mask=attention_mask,
            task=task,
            loss=loss,
            past_key_values=outputs['past_key_values']
        )

    def _calculate_loss(
        self,
        logits: torch.Tensor,
        task: str,
        input_labels: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        loss_fn_reduction: str = 'mean',
        nan_target_idx: int = -1
    ) -> Optional[torch.Tensor]:
        """Calculate loss for a specific task.
        
        Args:
            logits: Model logits
            task: Task type
            input_labels: Labels for language modeling tasks
            target: Target values for prediction tasks
            loss_fn_reduction: Reduction method for loss calculation
            nan_target_idx: Index for NaN targets in prediction tasks
            
        Returns:
            Calculated loss or None if inputs are missing
        """
        if task == 'lm':
            if input_labels is not None:
                batch_size, seq_len, vocab_size = logits.size()
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_labels = input_labels[:, 1:].contiguous()
                return F.cross_entropy(
                    shifted_logits.view(batch_size * (seq_len - 1), vocab_size),
                    shifted_labels.view(batch_size * (seq_len - 1)),
                    ignore_index=IGNORE_TOKEN_IDX,
                    reduction=loss_fn_reduction
                )
        elif task == 'mlm':
            if input_labels is not None:
                return F.cross_entropy(
                    logits, 
                    input_labels, 
                    ignore_index=IGNORE_TOKEN_IDX, 
                    reduction=loss_fn_reduction
                )
        elif task == 'prediction':
            if target is not None:
                if self.prediction_prediction_task_type == 'classification':
                    return F.binary_cross_entropy_with_logits(
                        logits[target != nan_target_idx].view(-1, ), 
                        target[target != nan_target_idx].view(-1, ), 
                        reduction=loss_fn_reduction
                    )
                elif self.prediction_prediction_task_type == 'regression':
                    return F.mse_loss(
                        logits.view(-1, ), 
                        target.view(-1, ), 
                        reduction=loss_fn_reduction
                    )
        return None
    
    @inference
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        _logits = self(input_ids=input_ids, attention_mask=attention_mask, task='prediction', **kwargs)['logits']
        if self.prediction_prediction_task_type == 'classification':
            return torch.sigmoid(_logits)
        elif self.prediction_prediction_task_type == 'regression':
            return _logits
        else:
            raise ValueError('Variable `downstream_task` must be either `classification` or `regression`.')
    
    @inference
    def generate(
        self,
        prefix_input_ids,
        num_tokens_to_generate,
        eos_token_id,
        pad_token_id,
        temperature=1.0,
        top_k=25,
        top_p=None,
        use_cache=True
    ):
        """Generate sequence ids by sampling from the model.

        Parameters
        ----------
        prefix_input_ids : torch.Tensor
            Initial sequence of token ids to condition on.
        num_tokens_to_generate : int
            Number of new tokens to generate after the prefix.
        eos_token_id : int
            Token id that marks end of sequence.
        pad_token_id : int
            Token id used for padding.
        temperature : float, optional
            Temperature for logits scaling, by default 1.0.
        top_k : int, optional
            Number of highest probability tokens to keep for top-k filtering, by default 25.
        top_p : float, optional
            Cumulative probability threshold for nucleus sampling, by default None.
        use_cache : bool, optional
            Whether to use KV caching for faster generation, by default True.

        Returns
        -------
        torch.Tensor
            Generated sequences including the prefix. Shape: (batch_size, sequence_length).
            The sequence_length will be <= prefix_length + num_tokens_to_generate,
            depending on when EOS tokens are generated.
        """
        # Initialize generation
        batch_size = prefix_input_ids.shape[0]
        device = prefix_input_ids.device
        prefix_len = prefix_input_ids.shape[1]
        
        # Pre-allocate output tensor
        output_ids = torch.full(
            (batch_size, prefix_len + num_tokens_to_generate),
            pad_token_id,
            dtype=torch.long,
            device=device
        )
        output_ids[:, :prefix_len] = prefix_input_ids

        # Initialize KV cache and EOS tracking
        past_key_values = None
        eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_len = prefix_len + num_tokens_to_generate

        # Generate tokens
        for pos_idx in range(prefix_len, prefix_len + num_tokens_to_generate):
            # Generate logits for all sequences
            next_token, past_key_values = self._generate_single_token(
                prefix_input_ids=output_ids[:, :pos_idx],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache,
                past_key_values=past_key_values
            )

            # For sequences that hit EOS, force next token to be PAD
            next_token = torch.where(eos_flags, pad_token_id, next_token)
            
            # Update output and EOS flags
            output_ids[:, pos_idx] = next_token
            eos_flags = eos_flags | (next_token == eos_token_id)

            # Early stopping if all sequences have hit EOS
            if eos_flags.all():
                generated_len = pos_idx + 1
                break

        # Ensure all sequences end with EOS (replacing last PAD if necessary)
        for sequence_idx, has_eos in enumerate(eos_flags):
            if not has_eos:
                output_ids[sequence_idx, generated_len - 1] = eos_token_id

        return output_ids[:, :generated_len]

    def _generate_single_token(self, prefix_input_ids, temperature, top_k, top_p, use_cache, past_key_values):
        """
        Generate a single token for each sequence in the batch, with optional KV caching.
        Args:
            prefix_input_ids: Input token ids
            temperature: Temperature for logits scaling
            top_k: Number of highest probability tokens to keep for top-k filtering
            top_p: Cumulative probability for nucleus sampling
            use_cache: Whether to use KV caching
            past_key_values: Cached key/value states from previous forward passes
        Returns:
            next_token: Generated token ids for each sequence in batch
            past_key_values: Updated KV cache if use_cache=True, else None
        """
        # Get logits using the same logic as generate_single_token_logits
        logits, past_key_values = self._generate_single_token_logits(
            prefix_input_ids=prefix_input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache,
            past_key_values=past_key_values
        )

        # Convert logits to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token, past_key_values

    def _generate_single_token_logits(self, prefix_input_ids, temperature, top_k, top_p, use_cache, past_key_values):
        """
        Generate logits for a single token for each sequence in the batch, with optional KV caching.
        Args:
            prefix_input_ids: Input token ids
            temperature: Temperature for logits scaling
            top_k: Number of highest probability tokens to keep for top-k filtering
            top_p: Cumulative probability for nucleus sampling
            use_cache: Whether to use KV caching
            past_key_values: Cached key/value states from previous forward passes
        Returns:
            logits: Output logits for next token prediction
            past_key_values: Updated KV cache if use_cache=True, else None
        """

        # Forward pass to get logits
        outputs = self(
            input_ids=prefix_input_ids,
            attention_mask=None,
            next_token_only=True,
            task='lm',
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache
        )
        logits = outputs['logits']
        past_key_values = outputs['past_key_values'] if use_cache else None

        # Scale logits by temperature
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove[..., 1:].size(-1) > 0:
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(dim=-1, index=indices_to_remove, value=-float('Inf'))

        return logits, past_key_values

    @classmethod
    def from_config(cls, config: ModelConfig, num_prediction_tasks: int = None, prediction_prediction_task_type: str = None, prediction_head_dropout_p: float = None):
        return cls(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_embedding_dim=config.hidden_embedding_dim,
            attention_dropout_p=config.attention_dropout_p,
            num_transformer_layers=config.num_transformer_layers,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            num_prediction_tasks=config.num_prediction_tasks if num_prediction_tasks is None else num_prediction_tasks,
            prediction_prediction_task_type=config.prediction_prediction_task_type if prediction_prediction_task_type is None else prediction_prediction_task_type,
            prediction_head_dropout_p=config.prediction_head_dropout_p if prediction_head_dropout_p is None else prediction_head_dropout_p
        )

    # def to_generator(self, tokenizer, batch_size, temperature, top_k, top_p = None, device = None) -> 'HyformerGeneratorWrapper':
    #     from hyformer.models.wrappers import HyformerGeneratorWrapper
    #     return HyformerGeneratorWrapper(self, tokenizer, batch_size, temperature, top_k, top_p, device)

    # def to_smiles_encoder(self, tokenizer, batch_size, device) -> EncoderWrapper:
    #     from hyformer.models.wrappers import HyformerSmilesEncoderWrapper
    #     return HyformerSmilesEncoderWrapper(self, tokenizer, batch_size, device)
