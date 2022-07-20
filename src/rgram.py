"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.utils import CfgNode as CN

# -----------------------------------------------------------------------------


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualBlock(nn.Module):
    def __init__(self, config: CN):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.n_embd, config.n_embd * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(config.n_embd * 4, config.n_embd)),
            ('dropout', nn.Dropout(config.resid_pdrop)),
        ]))
        self.ln_1 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor):
        x = x + self.mlp(self.ln_1(x))
        return x


class MergeBlock(nn.Module):
    def __init__(self, config: CN):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.n_embd * 2, config.n_embd * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(config.n_embd * 4, config.n_embd)),
            ('dropout', nn.Dropout(config.resid_pdrop)),
        ]))
        self.ln_1 = nn.LayerNorm(config.n_embd * 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x = torch.cat([x1, x2], dim=-1)
        x = x2 + self.mlp(self.ln_1(x))
        return x

class UnMergeBlock(nn.Module):
    def __init__(self, config: CN):
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.n_embd, config.n_embd * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(config.n_embd * 4, config.n_embd * 2)),
            ('dropout', nn.Dropout(config.resid_pdrop)),
        ]))
        self.ln_1 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.mlp(self.ln_1(x))
        return x


class NSP(nn.Module):
    def __init__(self, config: CN):
        super().__init__()
        self.n_embd = config.n_embd
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.resblock = ResidualBlock(config)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #self.resblock_out = ResidualBlock(config)
        self.mergeblocks = nn.ModuleList([MergeBlock(config) for _ in range(config.n_layer)])
        self.outprojs = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=True) for _ in range(config.n_layer)])
        self.ln_fs = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_layer)])
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) for _ in range(config.n_layer)])
        self.resblocks = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layer)])

    def forward(self, idx, targets=None):
        device = idx.device
        t = idx.size()

        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.drop(tok_emb)
        x = self.resblock(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        # If there are targets use them otherwise create fake last target
        if targets is not None:
            pred_targets = targets
        else:
            pred_targets = torch.cat([idx[1:], torch.empty_like(idx[:1]).fill_(-1)])

        # TODO: Double check that it's not forward leaking!
        for i, mergeblock in enumerate(self.mergeblocks):
            j = i + 1
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).view(-1)
            #_, idx_next = torch.topk(probs, k=1, dim=-1)
            idx_next = idx_next.view(-1)
            bool_indices = (idx_next == pred_targets).nonzero().view(-1)
            bool_indices = bool_indices[bool_indices < (t[0] - j)]  # Make sure we don't go out of bounds
            bool_indices_pj = bool_indices + j
            x1 = x[bool_indices]
            x2 = x[bool_indices_pj]
            x_merge = mergeblock(x1, x2)
            scatter_ix = bool_indices_pj.repeat_interleave(self.n_embd).view(-1, self.n_embd)
            torch.scatter(input=x, dim=0, index=scatter_ix, src=x_merge)
            x = self.outprojs[i](x)
            x = self.ln_fs[i](x)
            logits = self.lm_heads[i](x)

            if targets is not None:
                mse_loss = F.mse_loss(self.resblocks[i](x_merge[:-1]), x_merge[1:].detach())
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                if loss is not None:
                    loss = loss + mse_loss + ce_loss
                else:
                    loss = mse_loss + ce_loss

        return logits, loss


# -----------------------------------------------------------------------------


class Rgram(nn.Module):
    """ R-gram Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'rgram'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert (type_given and not params_given) or (not type_given and params_given) # exactly one of these
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                'rgram-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'rgram-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'rgram-nano':     dict(n_layer=1, n_head=3, n_embd=48),
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


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
        return optimizer

    def forward(self, idx, targets=None):
        idx = idx.view(-1)
        if targets is not None:
            targets = targets.view(-1)
        logits, loss = self.nsp(idx, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
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
            logits, _ = self(idx)
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
