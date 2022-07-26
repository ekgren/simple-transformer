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
from torch.nn import LayerNorm
from torch.nn import functional as F
from vector_quantize_pytorch import VectorQuantize

from src.utils import CfgNode as CN

# -----------------------------------------------------------------------------


#class LayerNorm(nn.LayerNorm):
#    """Subclass torch's LayerNorm to handle fp16."""
#    def forward(self, x: torch.Tensor):
#        orig_type = x.dtype
#        ret = super().forward(x.type(torch.float32))
#        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MergeBlock(nn.Module):
    def __init__(self, config: CN, level: int = 0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(config.n_embd * 2, config.n_mlp)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(config.n_mlp, config.n_embd)),
            ('dropout', nn.Dropout(config.resid_pdrop)),
        ]))
        self.ln = LayerNorm(config.n_embd * 2)
        self.shift = 2**level

    def forward(self,
                input: torch.Tensor,
                sample_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        print("In MergeBlock.forward", input.shape, sample_ids.shape)
        # Zero pad to the left in the seq dimension and remove the last shift elements
        print("input", input.shape, input.dtype, input)
        x_left_padded = F.pad(input, (0, 0, self.shift, -self.shift))
        print("x_left_padded", x_left_padded.shape, x_left_padded.dtype, x_left_padded)
        x_pairs = torch.cat([x_left_padded, input], dim=-1)
        print("x_pairs", x_pairs.shape)
        x_norm = self.ln(x_pairs)
        print("x_norm", x_norm.shape, x_norm.dtype, x_norm)
        x_merged = self.mlp(x_norm)
        print("x_merged", x_merged.shape)
        x_out = self.mask(input, x_merged, sample_ids) if sample_ids is not None else x_merged
        print("x_out", x_out.shape)
        return x_out

    def mask(self, input: torch.Tensor, x_merged: torch.Tensor, sample_ids: torch.Tensor) -> torch.Tensor:
        """ Mask the input to prevent out of bound memory accesses """
        # Zero pad to the left in the seq dimension and remove the last shift elements
        seq_ids_left_padded = F.pad(sample_ids, (self.shift, -self.shift))
        seq_ids_bool = (sample_ids == seq_ids_left_padded).unsqueeze(1).broadcast_to(input.shape)  # Do we need the unsqueeze?
        return torch.where(seq_ids_bool, x_merged, input)


class MergeBlocks(nn.Module):
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.mergeblocks = nn.ModuleList([MergeBlock(config, level=i) for i in range(config.n_layer)])
        self.lns = nn.ModuleList([LayerNorm(config.n_embd) for _ in range(config.n_layer)])
        # self.out_projs = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=True) for _ in range(config.n_layer)])

        # Test quantization again later
        # self.vq = VectorQuantize(dim=config.n_embd, codebook_size=config.vocab_size, decay=0.8, commitment_weight=1.)

    def forward(self,
                input: torch.Tensor,
                sample_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        print("In MergeBlocks", input.shape, sample_ids.shape)
        print("input", input.shape, input.dtype, input)
        for mergeblock, ln in zip(self.mergeblocks, self.lns):
            commit_loss = None
            input = ln(mergeblock(input, sample_ids) + input)  # merge -> residual -> layer norm
            # input = out_proj(input)  # linear projection
            return input, commit_loss


class NSP(nn.Module):
    """ Next Step Prediction """
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=-1)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_e = LayerNorm(config.n_embd)

        self.mergeblocks = nn.ModuleList([MergeBlocks(config) for _ in range(config.n_merges)])
        # self.temperatures = nn.ParameterList(
        #     OrderedDict(
        #     [('temp', nn.Parameter(torch.ones(1) * 3.14 / 2.)) for _ in range(config.n_merges)]
        #     ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self,
                idx: torch.Tensor,
                sample_ids: Optional[torch.Tensor] = None,
                pos_ids: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        print("In NSP.forward", idx.shape, sample_ids.shape, pos_ids.shape, targets.shape)
        pos_ids = pos_ids.view(1, -1)
        print("pos_ids", pos_ids.shape, pos_ids.dtype, pos_ids)
        device = idx.device
        #tok_emb = self.wte(idx)  # token embeddings of shape (b * t, n_embd)
        pos_emb = self.wpe(pos_ids)  # position embeddings of shape (b * t, n_pos_embd)
        x = pos_emb #+ torch.where(idx.view(-1, 1) > -1, tok_emb, pos_emb)
        print("x", x.shape, x.dtype, x)
        x = self.ln_e(x)
        x = self.drop(x)
        loss = None
        # for mergeblock, temp in zip(self.mergeblocks, self.temperatures):
        for mergeblock in self.mergeblocks:
            x, commit_loss = mergeblock(x, sample_ids)  # merge -> residual -> layer norm
            # logits = self.lm_head(x) / (1e-6 + (torch.sin(temp['temp']) + 1.)/2.)
            logits = self.lm_head(x)

            # Sample new tokens
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).view(-1)
            tok_emb = self.wte(idx)  # token embeddings of shape (b * t, n_embd)
            x = self.ln_e(tok_emb + x)
            x = self.drop(x)

            # If inference get loss
            if targets is not None:
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                loss = ce if loss is None else loss + ce

        return logits, loss


# -----------------------------------------------------------------------------


class RgramPos(nn.Module):
    """ R-gram Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_embd) must be given in the config
        C.model_type = 'rgrampos'
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
            # Which initialization to use?
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # torch.nn.init.xavier_uniform_(module.weight)
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
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, VectorQuantize)
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
                elif pn.endswith('temp'):
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

    def forward(self,
                input: torch.Tensor,
                targets: Optional[torch.Tensor] = None):  # Add type hint for output
        idx, sample_ids, pos_ids = input.unbind(0)
        print("idx:", idx.shape, idx)
        print("sample_ids:", sample_ids.shape, sample_ids)
        print("pos_ids:", pos_ids.shape, pos_ids)
        print("targets:", targets.shape, targets)
        logits, loss = self.nsp(idx=idx,
                                sample_ids=sample_ids,
                                pos_ids=pos_ids,
                                targets=targets)
        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx: torch.Tensor,
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 temperature: float = 1.0,
                 do_sample: bool = False,
                 top_k: Optional[int] = None) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx, device = idx.view(-1), idx.device
        idx_len = idx.size(0)
        if max_length is not None:
            seq_len = max_length
        elif max_new_tokens is not None:
            seq_len = idx_len + max_new_tokens
        else:
            seq_len = idx_len + 1

        if seq_len > idx_len:
            # pad idx with -1
            idx = torch.cat([idx,
                             torch.ones(seq_len - idx_len,
                                        dtype=idx.dtype,
                                        device=device) * -1],
                            dim=0).view(-1)

        sample_ids = idx.new_ones(seq_len).view(-1)
        pos_ids = torch.arange(seq_len, device=device).view(-1)
        input = torch.stack([idx, sample_ids, pos_ids], dim=0)
        # forward the model to get the logits for the index in the sequence
        logits, _ = self(input)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[:, logits < v[[-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)

        # append sampled index to the running sequence and continue
        # Think about this for language generation.
        # idx = torch.cat((idx, idx_next), dim=-1)

        return idx.unsqueeze(0)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear,)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()

        if isinstance(module, (nn.Embedding,)):
            module.weight.data = module.weight.data.half()

    model.apply(_convert_weights_to_fp16)
    print("Converted model weights to fp16")
