import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from hyformer.models.llama_backbone import LLAMABackbone
from hyformer.models.utils import ModelOutput
from hyformer.models.auto import AutoModel

from hyformer.utils.tokenizers.base import IGNORE_TOKEN_IDX
from hyformer.models.layers.prediction import PredictionHead
from hyformer.configs.model import ModelConfig

NAN_TARGET_IDX = -1


@AutoModel.register('Hyformer')
class Hyformer(LLAMABackbone):
    """ A joint transformer-based model for molecule generation and property prediction. Based on Llama backbone.
    """
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_embedding_dim: int, attention_dropout_p: float,
        num_transformer_layers: int, num_attention_heads: int, layer_norm_eps: float, num_prediction_tasks: int = None,
        prediction_task_type: str = None, prediction_head_dropout_p: float = None, init_weights: bool = True 
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
        prediction_task_type : str, optional
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
        self.prediction_task_type = prediction_task_type
        self.num_prediction_tasks = num_prediction_tasks
        if num_prediction_tasks is not None and prediction_task_type is not None:
            self.init_prediction_head(num_prediction_tasks=num_prediction_tasks, prediction_task_type=prediction_task_type, dropout_p=prediction_head_dropout_p)

    def init_prediction_head(self, num_prediction_tasks: int, prediction_task_type: str, dropout_p: float = None):
        self.prediction_head = PredictionHead(
            embedding_dim=self.embedding_dim,
            num_prediction_tasks=num_prediction_tasks,
            activation_fn='tanh' if prediction_task_type == 'classification' else 'gelu',
            dropout_p=dropout_p
        )

    def load_pretrained(self, state_dict: dict, discard_prediction_head: bool = False):
        if discard_prediction_head:
            removed_keys = [k for k in list(state_dict.keys()) if 'prediction_head' in k]
            if removed_keys:
                print(f"Warning: Removing prediction head: {removed_keys}")
                for k in removed_keys:
                    state_dict.pop(k)
        super().load_pretrained(state_dict=state_dict)
    
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
            logits = self.prediction_head(outputs['embeddings'][:, _cls_token_idx]) if self.prediction_head is not None else None
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
            loss=loss
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
                batch_size, seq_len, vocab_size = logits.size()
                return F.cross_entropy(
                    logits.view(batch_size * seq_len, vocab_size), 
                    input_labels.view(batch_size * seq_len), 
                    ignore_index=IGNORE_TOKEN_IDX, 
                    reduction=loss_fn_reduction
                )
        elif task == 'prediction':
            if target is not None:
                if self.prediction_task_type == 'classification':
                    _is_binary_classification = logits.shape[-1] == 1
                    valid_mask = target != nan_target_idx
                    
                    if _is_binary_classification:
                        valid_logits = logits[valid_mask].view(-1)
                        valid_targets = target[valid_mask].view(-1)
                        return F.binary_cross_entropy_with_logits(
                            valid_logits,
                            valid_targets,
                            reduction=loss_fn_reduction
                        )
                    else:
                        batch_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                        valid_logits = logits[batch_indices]
                        valid_targets = target[batch_indices]
                        return F.cross_entropy(
                            valid_logits,
                            valid_targets,
                            reduction=loss_fn_reduction
                        )
                elif self.prediction_task_type == 'regression':
                    return F.mse_loss(
                        logits.view(-1, ), 
                        target.view(-1, ), 
                        reduction=loss_fn_reduction
                    )
        return None
    
    @torch.inference_mode()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        _logits = self(input_ids=input_ids, attention_mask=attention_mask, task='prediction', **kwargs)['logits']
        if self.prediction_task_type == 'classification':
            return torch.sigmoid(_logits)
        elif self.prediction_task_type == 'regression':
            return _logits
        else:
            raise ValueError('Variable `downstream_task` must be either `classification` or `regression`.')
    
    @torch.inference_mode()
    def generate(
        self,
        prefix_input_ids,
        num_tokens_to_generate,
        eos_token_id,
        pad_token_id,
        temperature=1.0,
        top_k=None,
        top_p=None,
        use_cache=False,
        device=None
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
        
        self.init_cache(batch_size, prefix_len + num_tokens_to_generate)
        # Pre-allocate output tensor
        output_ids = torch.full(
            (batch_size, prefix_len + num_tokens_to_generate),
            pad_token_id,
            dtype=torch.long,
            device=device
        )
        output_ids[:, :prefix_len] = prefix_input_ids

        # Initialize KV cache and EOS tracking
        eos_flags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_len = prefix_len + num_tokens_to_generate

        # Generate tokens
        for pos_idx in range(prefix_len, prefix_len + num_tokens_to_generate):
            # Generate logits for all sequences
            prefix_input_ids = output_ids[:, [pos_idx]] if use_cache and pos_idx > prefix_len else output_ids[:, :pos_idx]
            next_token = self._generate_single_token(
                prefix_input_ids=prefix_input_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache
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
        self.clear_cache()
        return output_ids[:, :generated_len]

    @torch.inference_mode()
    def _generate_single_token(self, prefix_input_ids, temperature, top_k, top_p, use_cache):
        """
        Generate a single token for each sequence in the batch, with optional KV caching.
        Args:
            prefix_input_ids: Input token ids
            temperature: Temperature for logits scaling
            top_k: Number of highest probability tokens to keep for top-k filtering
            top_p: Cumulative probability for nucleus sampling
            use_cache: Whether to use KV caching
        Returns:
            next_token: Generated token ids for each sequence in batch
        """
        # Get logits using the same logic as generate_single_token_logits
        logits = self._generate_single_token_logits(
            prefix_input_ids=prefix_input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=use_cache
        )

        # Convert logits to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token

    def _generate_single_token_logits(self, prefix_input_ids, temperature, top_k, top_p, use_cache):
        """
        Generate logits for a single token for each sequence in the batch, with optional KV caching.
        Args:
            prefix_input_ids: Input token ids
            temperature: Temperature for logits scaling
            top_k: Number of highest probability tokens to keep for top-k filtering
            top_p: Cumulative probability for nucleus sampling
            use_cache: Whether to use KV caching
        Returns:
            logits: Output logits for next token prediction
        """

        # Forward pass to get logits
        outputs = self(
            input_ids=prefix_input_ids,
            attention_mask=None,
            next_token_only=True,
            task='lm',
            use_cache=use_cache
        )
        logits = outputs['logits']

        # Scale logits by temperature
        logits = logits / temperature

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = 0
            for idx in range(logits.size(0)):
                indices_to_remove = sorted_indices[idx, sorted_indices_to_remove[idx]]
                logits[idx, indices_to_remove] = -float('Inf')
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        return logits

    @classmethod
    def from_config(cls, config: ModelConfig, num_prediction_tasks: int = None, prediction_task_type: str = None, prediction_head_dropout_p: float = None):
        return cls(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_embedding_dim=config.hidden_embedding_dim,
            attention_dropout_p=config.attention_dropout_p,
            num_transformer_layers=config.num_transformer_layers,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            num_prediction_tasks=config.num_prediction_tasks if num_prediction_tasks is None else num_prediction_tasks,
            prediction_task_type=config.prediction_task_type if prediction_task_type is None else prediction_task_type,
            prediction_head_dropout_p=config.prediction_head_dropout_p if prediction_head_dropout_p is None else prediction_head_dropout_p
        )
    
    def to_generator(self, **kwargs) -> 'HyformerGeneratorWrapper':
        from hyformer.models.wrappers import HyformerGeneratorWrapper
        return HyformerGeneratorWrapper(self, **kwargs)

    def to_encoder(self, tokenizer, batch_size, device) -> 'HyformerEncoderWrapper':
        from hyformer.models.wrappers import HyformerEncoderWrapper
        return HyformerEncoderWrapper(self, tokenizer, batch_size, device)
