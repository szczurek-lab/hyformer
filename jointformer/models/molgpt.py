
import re
import math
import logging
import torch

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from jointformer.models.base import BaseModel, SmilesEncoder


logger = logging.getLogger(__name__)


class MolGPT(BaseModel, SmilesEncoder):
    def __init__(self) -> None:
        super().__init__()
        self._config = None
        self._model = None
        self._dataset = None
        self._tokenizer = None
        self._batch_size = None
        self._temperature = None
        self._top_k = None
        self._device = None
    
    def to_smiles_encoder(self, tokenizer=None, batch_size=64, device='cuda:0') -> SmilesEncoder:
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        return self

    def to_guacamole_generator(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, smiles: list[str]) -> np.ndarray:
        assert self._model is not None, "Model is not loaded"
        dataset = SmilesDataset(smiles)
        dataloader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=self._batch_size, num_workers=4)
        self._model.eval()
        self._model.to(self._device)
        embeddings = np.zeros((len(smiles), self._model.config.n_embd))
        for i, (x, _) in enumerate(tqdm(dataloader)):
            x = x.to(self._device)
            embeddings[i*self._batch_size:(i+1)*self._batch_size] = self._model(x)['embeddings'].mean(1).cpu().numpy()
        return embeddings

    def load_pretrained(self, filename, *args, **kwargs):
        self._vocab_size = 94  # model default Guacamol vocab size
        self._max_length = 100  # model default Guacamol max length
        self._scaffold = False  # model default Guacamol scaffold
        self._scaffold_max_len = 1  # model default Guacamol scaffold max length

        self._config = GPTConfig(
            self._vocab_size, self._max_length, num_props=0, n_layer=8, n_head=8, n_embd=256,
            scaffold=self._scaffold, scaffold_maxlen=self._scaffold_max_len, lstm=False, lstm_layers=False)
        self._model = GPT(self._config)
        _ckpt = torch.load(filename, map_location='cpu')
        self._model.load_state_dict(self._filter_checkpoint(_ckpt))
        del _ckpt
    
    def _filter_checkpoint(self, ckpt: dict) -> dict:
        model_params = [param for param in self._model.state_dict().keys()]
        filtered_checkpoint = {}
        for key, value in ckpt.items():
            modified_key = '_model.' + key
            if modified_key in model_params:
                filtered_checkpoint[modified_key] = value
            if key in model_params:
                filtered_checkpoint[key] = value
            else:
                print(f"Removed key: {key} from checkpoint; not found in model or size mismatch.")
        return filtered_checkpoint

    @classmethod
    def from_config(cls, config):
        return cls()


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        num = int(bool(config.num_props)) + int(config.scaffold_maxlen)   #int(config.lstm_layers)    #  int(config.scaffold) 
        # num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)
     
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        if config.lstm:
            self.lstm = nn.LSTM(input_size = config.n_embd, hidden_size = config.n_embd, num_layers = config.lstm_layers, dropout = 0.3, bidirectional = False)
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def forward(self, idx, targets=None, prop = None, scaffold = None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.config.num_props:
            assert prop.size(-1) == self.config.num_props, "Num_props should be equal to last dim of property vector"           

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        type_embeddings = self.type_emb(torch.ones((b,t), dtype = torch.long, device = idx.device))
        x = self.drop(token_embeddings + position_embeddings + type_embeddings)

        if self.config.num_props:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))
            if prop.ndim == 2:
                p = self.prop_nn(prop.unsqueeze(1))    # for single property
            else:
                p = self.prop_nn(prop)    # for multiproperty
            p += type_embd
            x = torch.cat([p, x], 1)

        if self.config.scaffold:
            type_embd = self.type_emb(torch.zeros((b, 1), dtype = torch.long, device = idx.device))

            scaffold_embeds = self.tok_emb(scaffold)     # .mean(1, keepdim = True)
            if self.config.lstm:
                scaffold_embeds = self.lstm(scaffold_embeds.permute(1,0,2))[1][0]
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], 2, self.config.n_embd).mean(2)
                scaffold_embeds = scaffold_embeds.permute(1,0,2)   # mean(0, keepdim = True)
                # scaffold_embeds = scaffold_embeds.reshape(self.config.lstm_layers, 1, -1, self.config.n_embd)[-1].permute(1,0,2)
                # scaffold_embeds = scaffold_embeds.reshape(scaffold_embeds.shape[1], scaffold_embeds.shape[0], self.config.n_embd)
            scaffold_embeds += type_embd
            x = torch.cat([scaffold_embeds, x], 1)

        # x = self.blocks(x)
        attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        # print(logits.shape)
        if self.config.num_props and self.config.scaffold:
            num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)
        elif self.config.num_props:
            num = int(bool(self.config.num_props))
        elif self.config.scaffold:
            num = int(self.config.scaffold_maxlen) 
        else:
            num = 0

        logits = logits[:, num:, :]


        # if self.config.num_props or self.config.scaffold:

        #     num = int(bool(self.config.num_props)) + int(self.config.scaffold_maxlen)  #int(self.config.lstm_layers)   # int(self.config.scaffold)      # int(self.config.scaffold)
            

        # print(logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return {'embeddings': x, 'logits': logits, 'loss': loss, 'attn_maps': attn_maps}  # (num_layers, batch_size, num_heads, max_seq_len, max_seq_len)


class SmilesDataset(Dataset):

    CHARACTERS = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']
    PATTERN =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    MAX_LENGTH = 100

    def __init__(self, data):
        chars = sorted(list(set(self.CHARACTERS)))
        _, self._vocab_size = len(data), len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = self.MAX_LENGTH
        self.regex = re.compile(self.PATTERN)
        self.data = self._to_molgpt_input_data(data)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        
        smiles += str('<')*(self.max_len - len(self.regex.findall(smiles)))
        smiles = self.regex.findall(smiles)
        dix = [self.stoi[s] for s in smiles]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

    def _to_molgpt_input_data(self, data: list[str]) -> list[str]:
        trimmed_data = [smiles if len(self.regex.findall(smiles)) <= self.max_len else smiles[:self.max_len] for smiles in data]
        if len(trimmed_data) < len(data):
            logger.warning(f"Trimmed {len(data) - len(trimmed_data)} examples to max length of {self.max_len}.")
        data_filtered = []
        num_examples_filtered = 0
        for smiles in tqdm(trimmed_data, desc='Filtering data'):
            smiles = smiles.strip()
            smiles_filtered = ''
            for token in self.regex.findall(smiles):
                if token in self.stoi:
                    smiles_filtered += token
                else:
                    num_examples_filtered += 1
                    token_with_charge_removed = token.replace('+', '').replace('-', '').replace('[', '').replace(']', '')
                    if token_with_charge_removed in self.stoi:
                        smiles_filtered += token_with_charge_removed
            data_filtered.append(smiles_filtered)
        if num_examples_filtered > 0:
            logger.warning(f"Filtered {num_examples_filtered} examples due to unknown characters.")
        input_data = [i + str('<')*(self.max_len - len(self.regex.findall(i.strip()))) for i in data_filtered]
        return input_data
    
    @property
    def vocab_size(self):
        return self._vocab_size
    