import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional
from jointformer.models.layers.prediction import DownstreamPredictionHead
from jointformer.models.gpt import LayerNorm, Block


class FancyModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.max_seq_len = config.max_seq_len

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            type_emb = nn.Embedding(2, config.n_embd) if config.num_props is not None else nn.Identity() ,  # 0 for tokens, 1 for types
            wpe = nn.Embedding(config.block_size, config.n_embd),
            prop_emb = nn.Linear(config.num_props, config.n_embd) if config.num_props is not None else nn.Identity(),  # property embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.property_head = nn.Linear(config.n_embd, config.num_props, bias=False) if config.num_props is not None else nn.Identity()
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, input_labels: torch.Tensor = None, attention_mask: torch.Tensor = None,
                properties: torch.Tensor = None, next_token_only: Optional[bool] = False, **kwargs):

        device = input_ids.device
        batch_size, sequence_length = input_ids.size()
        assert sequence_length <= self.config.block_size, f"Cannot forward sequence of length {sequence_length}, block size is only {self.config.block_size}"
        
        # Embedding of the sequence tokens
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos = torch.arange(0, sequence_length, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        type_tokens = torch.zeros(sequence_length, dtype=torch.long, device=device)
        type_emb = self.transformer.type_emb(type_tokens) if self.config.num_props is not None else 0
        token_embeddings = tok_emb + pos_emb + type_emb
        property_embeddings = None

        # Embedding of the property tokens
        if properties is not None:
            prop_emb = self.transformer.prop_emb(properties).unsqueeze(1) # property embeddings of shape (b, 1, n_embd)
            type_tokens = torch.ones(1, dtype=torch.long, device=device)
            type_emb = self.transformer.type_emb(type_tokens)
            property_embeddings = prop_emb + type_emb

        embeddings = torch.cat([token_embeddings, property_embeddings], dim=1) if property_embeddings is not None else token_embeddings
        x = self.transformer.drop(embeddings)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        loss = None
        if next_token_only:
            logits = self.lm_head(x[:, [-1], :])
            return {'logits_generation': logits}
        else:
            logits = self.lm_head(x[:, :sequence_length, :])
            
        if input_labels is not None:
            input_labels = input_labels[:, 1:].contiguous()
            logits = logits[:, :-1].contiguous()
            batch_size, sequence_length = input_labels.size()
            loss = F.cross_entropy(logits.view(batch_size * sequence_length, -1), input_labels.view(batch_size * sequence_length), ignore_index=-100)
        
        if properties is not None and loss is not None:
            property_prediction = self.property_head(x[:, -1, :])
            loss += F.mse_loss(property_prediction, properties) 

        return {'token_embeddings': embeddings, 'embeddings': x, 'logits': logits, 'loss': loss}

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def get_loss(self, input_ids: torch.Tensor, input_labels: torch.Tensor = None, attention_mask: torch.Tensor = None,
                properties: torch.Tensor = None, next_token_only: Optional[bool] = False, **kwargs):
        return self(input_ids=input_ids, input_labels=input_labels, properties=properties)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, tokenizer, batch_size, temperature, top_k, device):
        
        assert hasattr(tokenizer, 'generation_prefix'), "Tokenizer must have a `generation_prefix` attribute."
        
        prefix_token = tokenizer.generation_prefix 
        prefix_token_length = 1 if isinstance(prefix_token, int) else len(prefix_token)
        eos_token_id = tokenizer.sep_token_id
        pad_token_id = tokenizer.pad_token_id

        
        prefix = torch.tensor(prefix_token, device=device).long().unsqueeze(0).expand(batch_size, -1)
        generated_tokens = self._generate(prefix, tokenizer.max_molecule_length - prefix_token_length, temperature, top_k, eos_token_id, pad_token_id)
        return generated_tokens
    
    def _generate(self, idx, max_new_tokens, temperature, top_k, eos_token_id, pad_token_id):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        eos_flag = torch.zeros(size=(idx.size(0), 1), dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            if eos_token_id:
                is_end = torch.logical_or(idx[:, [-1]] == eos_token_id, idx[:, [-1]] == pad_token_id)
                eos_flag = torch.logical_or(eos_flag, is_end)
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            outputs = self(input_ids=idx_cond, attention_mask=None, next_token_only=True, task='generation')
            logits = outputs['logits_generation']

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.where(eos_flag, torch.ones_like(idx_next) * pad_token_id, idx_next)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    @classmethod
    def from_config(cls, config):
        config.block_size = config.max_seq_len
        config.n_embd = config.embedding_dim
        config.n_layer = config.num_layers
        config.n_head = config.num_heads
        config.num_props = config.num_physchem_tasks
        return cls(
            config=config
        )


class FancyModelForDownstreamPrediction(FancyModel):

    def __init__(self, config):
            
        super().__init__(config)
        self.prediction_task_type = config.downstream_task
        self.num_prediction_tasks = config.num_tasks
        self.downstream_prediction_task_head = DownstreamPredictionHead(
            config.n_embd, 2 if config.downstream_task == 'classification' and config.num_tasks == 1 else config.num_tasks, config.hidden_dim)

    def forward(self, input_ids: torch.Tensor, input_labels: torch.Tensor = None, attention_mask: torch.Tensor = None,
                properties: torch.Tensor = None, next_token_only: Optional[bool] = False, **kwargs):
        outputs = super().forward(input_ids=input_ids, input_labels=input_labels, attention_mask=attention_mask,
                                  properties=properties, next_token_only=next_token_only, **kwargs)
        outputs['logits_prediction'] = self.downstream_prediction_task_head(outputs['embeddings'][:, -1, :])
        return outputs

    def get_loss(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, properties: torch.Tensor, **kwargs):
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.prediction_task_type == 'classification':
            if self.num_prediction_tasks == 1:
                outputs["loss"] = F.cross_entropy(outputs["logits_prediction"], properties, reduction='mean')
            elif self.num_prediction_tasks > 1:
                outputs["loss"] = F.binary_cross_entropy_with_logits(outputs["logits_prediction"], properties, reduction='mean')
            else:
                raise ValueError('Variable `num_prediction_tasks` must be greater than 0.')
            
        elif self.prediction_task_type == 'regression':
            outputs["loss"] = F.mse_loss(outputs["logits_prediction"].flatten(), properties.flatten(), 'mean')
        
        else:
            raise ValueError('Variable `downstream_task` must be either `classification` or `regression`.')
        
        return outputs
    
    @classmethod
    def from_config(cls, config, downstream_task, num_tasks, hidden_dim):
        config.block_size = config.max_seq_len
        config.n_embd = config.embedding_dim
        config.n_layer = config.num_layers
        config.n_head = config.num_heads
        config.num_props = config.num_physchem_tasks
        config.downstream_task = downstream_task
        config.num_tasks = num_tasks
        config.hidden_dim = hidden_dim
        return cls(
            config=config
        )
