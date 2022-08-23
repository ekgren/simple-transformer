"""
Full definition of an Rgram model

References:
1) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
from collections import OrderedDict
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils import CfgNode as CN

# -----------------------------------------------------------------------------


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MergeBlock(nn.Module):
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.n_embd * 3, config.n_mlp)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(config.n_mlp, config.n_embd)),
            ('dropout', nn.Dropout(config.resid_pdrop)),
        ]))
        self.ln = LayerNorm(config.n_embd * 3)

    # TODO: Come up with a clear explanatory name for seq_ids
    def forward(self, input: torch.Tensor, lvl_emb: torch.Tensor, shift: int, seq_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Zero pad to the left in the seq dimension and remove the last shift elements
        x_left_padded = F.pad(input, (0, 0, shift, -shift))
        x_pairs = torch.cat([x_left_padded, input, lvl_emb], dim=-1)
        x_merged = self.mlp(self.ln(x_pairs))
        x_out = self.mask(input, x_merged, seq_ids) if seq_ids is not None else x_merged
        return x_out

    # TODO: Come up with a clear explanatory name for seq_ids
    def mask(self, input: torch.Tensor, x_merged: torch.Tensor, shift: int, seq_ids: torch.Tensor) -> torch.Tensor:
        """ Mask the input to prevent out of bound memory accesses """
        # Zero pad to the left in the seq dimension and remove the last shift elements
        seq_ids_left_padded = F.pad(seq_ids, (shift, -shift))
        seq_ids_bool = (seq_ids == seq_ids_left_padded).unsqueeze(1).broadcast_to(input.shape)  # Do we need the unsqueeze?
        return torch.where(seq_ids_bool, x_merged, input)


class MergeBlocks(nn.Module):
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.n_layer = config.n_layer
        self.mergeblock = MergeBlock(config)
        self.ln = LayerNorm(config.n_embd)
        self.le = nn.Embedding(config.n_layer, config.n_embd)

    def forward(self, input: torch.Tensor, seq_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i in torch.arange(self.n_layer).to(input.device):
            le = self.le(i)
            input = self.ln(self.mergeblock(input, le, shift=i.item()+1, seq_ids=seq_ids) + input)  # merge -> residual -> layer norm
            return input


class NSP(nn.Module):
    """ Next Step Prediction """
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_e = LayerNorm(config.n_embd)

        self.mergeblocks = nn.ModuleList([MergeBlocks(config) for _ in range(config.n_merges)])

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # TODO: Come up with a clear explanatory name for seq_ids
    def forward(self, idx: torch.Tensor, seq_ids: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        tok_emb = self.wte(idx)  # token embeddings of shape (b * t, n_embd)
        x = self.ln_e(tok_emb)
        x = self.drop(x)
        for mergeblock in self.mergeblocks:
            x = mergeblock(x, seq_ids)  # merge -> residual -> layer norm
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss


# -----------------------------------------------------------------------------


class Rgram(nn.Module):
    """ R-gram Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_embd) must be given in the config
        C.model_type = 'rgram'
        C.n_layer = None
        C.n_embd = None
        C.n_mlp = None
        C.n_merges = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config: CN) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_embd is not None])
        assert (type_given and not params_given) or (not type_given and params_given) # exactly one of these
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                'rgram-mini':     dict(n_layer=6, n_embd=192, n_mlp=4*192, n_merges=1),
                'rgram-micro':    dict(n_layer=4, n_embd=128, n_mlp=4*128, n_merges=1),
                'rgram-nano':     dict(n_layer=2, n_embd=48, n_mlp=4*48, n_merges=1),
            }[config.model_type])

        self.nsp = NSP(config)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.nsp.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            #torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config: CN):  # Add type hint for output
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

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
        # optimizer = bnb.optim.Adam8bit(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, input, targets: Optional[torch.Tensor] = None):  # Add type hint for output
        idx, sample_ids, pos_ids = input.unbind(0)
        logits, loss = self.nsp(idx, sample_ids, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 do_sample: bool = False, top_k: Optional[int] = None) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = idx.view(-1)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            #idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            seq_ids = idx.new_ones(idx.shape)
            logits, _ = self(torch.stack((idx, seq_ids)))
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[-1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[[-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=-1)

        return idx.unsqueeze(0)