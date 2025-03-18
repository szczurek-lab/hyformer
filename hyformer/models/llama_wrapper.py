import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast

from hyformer.models.trainable import TrainableModel
from hyformer.models.base import BaseModel, SmilesEncoder
from hyformer.models.layers.prediction import RegressionHead, ClassificationHead
from hyformer.models.utils import ModelOutput

class HyformerLlama(BaseModel, TrainableModel):
    """
    Hyformer model that uses LLaMA as a backbone.
    This adapter preserves Hyformer's configuration system while using LLaMA's architecture.
    """
    
    def __init__(
            self,
            vocab_size: int,
            max_sequence_length: int,
            embedding_dim: int,
            hidden_embedding_dim: int,
            attention_dropout: float,
            hidden_dropout: float,
            num_layers: int,
            use_bias: bool,
            num_attention_heads: int,
            layer_norm_eps: float,
            num_physchem_tasks: int = 200,
            llama_model_name: str = "meta-llama/Llama-3-8B",
            predictor_dropout: float = 0.0,
            flash_attention: bool = True,
            init_weights: bool = True,
            prediction_task_type: str = "regression",
            num_prediction_tasks: int = 1,
            classifier_dropout: float = 0.0
    ):
        super().__init__()
        
        # Store Hyformer configuration parameters
        self.vocab_size = vocab_size
        self.max_seq_len = max_sequence_length
        self.embedding_dim = embedding_dim
        self.embedding_hidden_dim = hidden_embedding_dim
        self.attention_dropout = attention_dropout
        self.feed_forward_dropout = hidden_dropout
        self.num_layers = num_layers
        self.bias = use_bias
        self.num_heads = num_attention_heads
        self.layer_norm_eps = layer_norm_eps
        self.num_physchem_tasks = num_physchem_tasks
        self.llama_model_name = llama_model_name
        self.pooler_dropout = predictor_dropout
        self.flash_attention = flash_attention
        self.prediction_task_type = prediction_task_type
        self.num_prediction_tasks = num_prediction_tasks
        
        # Initialize LLaMA configuration
        # We'll use LLaMA's native dimensions but adapt our configuration parameters where possible
        self.llama_config = LlamaConfig.from_pretrained(
            llama_model_name,
            vocab_size=vocab_size,  # Use our vocabulary size
            max_position_embeddings=max_sequence_length,  # Use our max sequence length
            hidden_size=embedding_dim,  # Map to our embedding dimension if possible
            intermediate_size=hidden_embedding_dim,  # Map to our hidden dimension
            num_hidden_layers=num_layers,  # Use our number of layers
            num_attention_heads=num_attention_heads,  # Use our number of attention heads
            rms_norm_eps=layer_norm_eps,  # Use our layer norm epsilon
            attention_dropout=attention_dropout,  # Use our attention dropout
            hidden_dropout=hidden_dropout,  # Use our feed-forward dropout
            use_cache=True,
            pad_token_id=0,  # Adjust based on your tokenizer
        )
        
        # Initialize LLaMA model
        try:
            # Try to load pretrained model
            self.backbone = LlamaModel.from_pretrained(llama_model_name, config=self.llama_config)
        except Exception as e:
            print(f"Could not load pretrained LLaMA model: {e}")
            print("Initializing LLaMA model from scratch")
            self.backbone = LlamaModel(self.llama_config)
        
        # Task-specific heads
        self.lm_head = nn.Linear(self.llama_config.hidden_size, vocab_size, bias=False)
        
        # Prediction head based on task type
        if prediction_task_type == "regression":
            self.prediction_head = RegressionHead(
                embedding_dim=self.llama_config.hidden_size,
                prediction_hidden_dim=hidden_embedding_dim // 4,
                output_dim=num_prediction_tasks
            )
        else:
            self.prediction_head = ClassificationHead(
                embedding_dim=self.llama_config.hidden_size,
                prediction_hidden_dim=hidden_embedding_dim // 4,
                output_dim=num_prediction_tasks
            )
        
        # Physicochemical property prediction head
        self.physchem_head = RegressionHead(
            embedding_dim=self.llama_config.hidden_size,
            prediction_hidden_dim=hidden_embedding_dim // 4,
            output_dim=num_physchem_tasks
        )
        
        # Pooler for sequence-level tasks
        self.pooler_dropout = nn.Dropout(predictor_dropout)
        
        # Initialize weights if needed
        if init_weights:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the task-specific heads."""
        for module in [self.lm_head, self.prediction_head, self.physchem_head]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            input_labels: torch.Tensor = None,
            properties: torch.Tensor = None,
            task: str = 'generation',
            next_token_only: bool = False,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            input_labels: Labels for language modeling
            properties: Properties for prediction tasks
            task: Task to perform ('generation', 'prediction', 'physchem')
            next_token_only: Whether to return only the next token prediction
            
        Returns:
            Dictionary with task-specific outputs
        """
        # Get LLaMA backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Task-specific processing
        if task == 'generation':
            return self.get_loss_lm(hidden_states, input_labels, next_token_only)
        elif task == 'prediction':
            return self.get_loss_prediction(hidden_states, attention_mask, properties)
        elif task == 'physchem':
            return self.get_loss_physchem(hidden_states, attention_mask, properties)
        else:
            raise ValueError(f"Task {task} not supported")
    
    def get_loss_lm(
            self,
            hidden_states: torch.Tensor,
            input_labels: torch.Tensor = None,
            next_token_only: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get language modeling loss."""
        if next_token_only:
            logits = self.lm_head(hidden_states[:, [-1], :])
            return {'logits_generation': logits}
        else:
            logits = self.lm_head(hidden_states)
            
        loss = None
        if input_labels is not None:
            input_labels = input_labels[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            batch_size, sequence_length = input_labels.size()
            loss = F.cross_entropy(
                logits.view(batch_size * sequence_length, -1),
                input_labels.view(batch_size * sequence_length),
                ignore_index=-100
            )
        
        return {'token_embeddings': hidden_states, 'embeddings': hidden_states, 'logits': logits, 'loss': loss}
    
    def get_loss_prediction(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            properties: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Get prediction loss."""
        # Global pooling
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Mean pooling
            pooled = hidden_states.mean(dim=1)
        
        pooled = self.pooler_dropout(pooled)
        logits = self.prediction_head(pooled)
        
        outputs = {"logits_prediction": logits}
        
        if properties is not None:
            if self.prediction_task_type == 'classification':
                if self.num_prediction_tasks == 1:
                    outputs["loss"] = F.binary_cross_entropy_with_logits(
                        logits.view(-1), properties.view(-1), reduction='mean'
                    )
                else:
                    _logits = logits[properties != -1].view(-1)
                    _properties = properties[properties != -1].view(-1)
                    outputs["loss"] = F.binary_cross_entropy_with_logits(
                        _logits, _properties, reduction='mean'
                    )
            elif self.prediction_task_type == 'regression':
                outputs["loss"] = F.mse_loss(logits.flatten(), properties.flatten(), 'mean')
        
        return outputs
    
    def get_loss_physchem(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            properties: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Get physicochemical property prediction loss."""
        # Global pooling
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Mean pooling
            pooled = hidden_states.mean(dim=1)
        
        logits = self.physchem_head(pooled)
        
        outputs = {"logits_physchem": logits}
        
        if properties is not None:
            outputs["loss"] = F.mse_loss(logits, properties, 'mean')
        
        return outputs
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Make predictions for downstream tasks."""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task='prediction',
            **kwargs
        )
        return outputs["logits_prediction"]
    
    def generate(
            self,
            tokenizer,
            batch_size: int = 16,
            temperature: float = 1.0,
            top_k: int = 40,
            max_length: int = 100,
            device: torch.device = None,
            **kwargs
    ) -> torch.Tensor:
        """Generate sequences using the LLaMA model."""
        # Create a LLaMA for causal LM model for generation
        lm_model = LlamaForCausalLM(self.llama_config)
        lm_model.model = self.backbone  # Use our backbone
        lm_model.lm_head = self.lm_head  # Use our LM head
        
        # Generate sequences
        input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * tokenizer.bos_token_id
        
        generated_ids = lm_model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs
        )
        
        return generated_ids
    
    def configure_optimizers(
            self,
            weight_decay: float = 0.01,
            learning_rate: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.999),
            device: torch.device = None
    ) -> torch.optim.Optimizer:
        """Configure optimizer with weight decay."""
        # Separate parameters that should have weight decay from those that shouldn't
        decay_params = []
        nodecay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if len(param.shape) > 1:  # Apply weight decay to matrices but not biases/norms
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        
        # Use AdamW optimizer with fused implementation if available
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device is not None and device.type == 'cuda'
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused if use_fused else False
        )
        
        return optimizer
    
    def to_guacamole_generator(self, tokenizer, batch_size, temperature, top_k, device) -> 'DistributionMatchingGenerator':
        """Create a generator for GuacaMol distribution matching tasks."""
        from hyformer.models.wrappers import HyformerSmilesGeneratorWrapper
        return HyformerSmilesGeneratorWrapper(self, tokenizer, batch_size, temperature, top_k, device)
    
    def to_smiles_encoder(self, tokenizer, batch_size, device) -> SmilesEncoder:
        """Create a SMILES encoder."""
        from hyformer.models.wrappers import HyformerSmilesEncoderWrapper
        return HyformerSmilesEncoderWrapper(self, tokenizer, batch_size, device)
    
    def to_downstream_predictive_model(self, task_type, num_tasks, prediction_hidden_dim):
        """Create a downstream predictive model."""
        from hyformer.models.wrappers import DownstreamPredictiveModelWrapper
        return DownstreamPredictiveModelWrapper(self, task_type, num_tasks, prediction_hidden_dim)
    
    def load_pretrained(self, filename, device='cpu'):
        """Load pretrained weights."""
        state_dict = torch.load(filename, map_location=device, weights_only=True)
        
        # Handle model key in state dict
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Handle compiled model artifacts
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        # Load state dict with strict=False to allow for missing keys
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded pretrained weights from {filename}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        return self
    
    @classmethod
    def from_config(cls, config):
        """Create a model from configuration."""
        return cls(
            vocab_size=config.vocab_size,
            max_sequence_length=config.max_sequence_length,
            embedding_dim=config.embedding_dim,
            hidden_embedding_dim=config.hidden_embedding_dim,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            num_layers=config.num_layers,
            use_bias=config.use_bias,
            num_attention_heads=config.num_attention_heads,
            layer_norm_eps=config.layer_norm_eps,
            num_physchem_tasks=config.num_physchem_tasks if hasattr(config, 'num_physchem_tasks') else 200,
            llama_model_name=config.llama_model_name if hasattr(config, 'llama_model_name') else "meta-llama/Llama-3-8B",
            predictor_dropout=config.predictor_dropout if hasattr(config, 'predictor_dropout') else 0.0,
            flash_attention=config.flash_attention if hasattr(config, 'flash_attention') else True,
            prediction_task_type=config.prediction_task_type if hasattr(config, 'prediction_task_type') else "regression",
            num_prediction_tasks=config.num_prediction_tasks if hasattr(config, 'num_prediction_tasks') else 1,
            classifier_dropout=config.classifier_dropout if hasattr(config, 'classifier_dropout') else 0.0
        )


class HyformerLlamaForDownstreamPrediction(HyformerLlama):
    """Hyformer with LLaMA backbone for downstream prediction tasks."""
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            properties: torch.Tensor = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass focused on prediction tasks."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        return self.get_loss_prediction(hidden_states, attention_mask, properties)


class HyformerLlamaForDownstreamPredictionDeep(HyformerLlama):
    """Hyformer with LLaMA backbone for downstream prediction tasks with deeper prediction head."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Replace the prediction head with a deeper one
        hidden_dim = self.embedding_hidden_dim // 2
        
        if self.prediction_task_type == "regression":
            self.prediction_head = nn.Sequential(
                nn.Linear(self.llama_config.hidden_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.pooler_dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.pooler_dropout),
                nn.Linear(hidden_dim // 2, self.num_prediction_tasks)
            )
        else:
            self.prediction_head = nn.Sequential(
                nn.Linear(self.llama_config.hidden_size, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.pooler_dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.pooler_dropout),
                nn.Linear(hidden_dim // 2, self.num_prediction_tasks)
            )
    
    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            properties: torch.Tensor = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass focused on prediction tasks."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs
        )
        
        hidden_states = outputs.last_hidden_state
        return self.get_loss_prediction(hidden_states, attention_mask, properties) 