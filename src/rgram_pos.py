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
from vector_quantize_pytorch import VectorQuantize

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

    # TODO: Come up with a clear explanatory name for seq_ids
    def forward(self, input: torch.Tensor, seq_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Zero pad to the left in the seq dimension and remove the last shift elements
        x_left_padded = F.pad(input, (0, 0, self.shift, -self.shift))
        x_pairs = torch.cat([x_left_padded, input], dim=-1)
        x_merged = self.mlp(self.ln(x_pairs))
        x_out = self.mask(input, x_merged, seq_ids) if seq_ids is not None else x_merged
        return x_out

    # TODO: Come up with a clear explanatory name for seq_ids
    def mask(self, input: torch.Tensor, x_merged: torch.Tensor, seq_ids: torch.Tensor) -> torch.Tensor:
        """ Mask the input to prevent out of bound memory accesses """
        # Zero pad to the left in the seq dimension and remove the last shift elements
        seq_ids_left_padded = F.pad(seq_ids, (self.shift, -self.shift))
        seq_ids_bool = (seq_ids == seq_ids_left_padded).unsqueeze(1).broadcast_to(input.shape)  # Do we need the unsqueeze?
        return torch.where(seq_ids_bool, x_merged, input)


class MergeBlocks(nn.Module):
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.mergeblocks = nn.ModuleList([MergeBlock(config, level=i) for i in range(config.n_layer)])
        self.lns = nn.ModuleList([LayerNorm(config.n_embd) for _ in range(config.n_layer)])
        self.vq = VectorQuantize(dim=config.n_embd, codebook_size=config.vocab_size, decay=0.8, commitment_weight=1.)

    def forward(self, input: torch.Tensor, seq_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        for ln, mergeblock in zip(self.lns, self.mergeblocks):
            quantized, indices, commit_loss = self.vq(input)  # (1, 1024, 256), (1, 1024), (1)
            input = ln(mergeblock(input, seq_ids) + quantized)  # merge -> residual -> layer norm
            return input, commit_loss


class NSP(nn.Module):
    """ Next Step Prediction """
    def __init__(self, config: CN) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # self.wte = bnb.nn.StableEmbedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.ln_e = LayerNorm(config.n_embd)

        self.mergeblocks = nn.ModuleList([MergeBlocks(config) for _ in range(config.n_merges)])

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        #self.lm_head.weight.data = self.wte.weight.data

    # TODO: Come up with a clear explanatory name for seq_ids
    def forward(self, idx: torch.Tensor, seq_ids: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = idx.device
        #rand_idx = torch.randint(0, self.vocab_size, (self.block_size,), device=device).view(-1)
        #rand_idx[0] = 256
        pos = torch.arange(0, self.block_size, dtype=torch.long, device=device).view(-1)
        #tok_emb = self.wte(rand_idx)  # token embeddings of shape (b * t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (b * t, n_pos_embd)
        x = self.ln_e(pos_emb)
        x = self.drop(x)
        loss = None
        for mergeblock in self.mergeblocks:
            x, commit_loss = mergeblock(x, seq_ids)  # merge -> residual -> layer norm
            loss = commit_loss if loss is None else loss + commit_loss
            with torch.no_grad():
                logits = self.lm_head(x)
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).view(-1)
            tok_emb = self.wte(idx)  # token embeddings of shape (b * t, n_embd)
            x = self.ln_e(tok_emb + x)
            x = self.drop(x)
        logits = self.lm_head(x)
        #loss = None
        if targets is not None:
            if logits.shape[0] != targets.shape[0]:
                logits = logits[:targets.shape[0], :]
            loss = loss + F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

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
            #torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        #elif isinstance(module, VectorQuantize):
        #    module.weight = self.nsp.wte.weight

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
        whitelist_weight_modules = (torch.nn.Linear, nn.MultiheadAttention, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm, VectorQuantize)
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

    def forward(self, idx, targets: Optional[torch.Tensor] = None):  # Add type hint for output
        idx, seq_ids = idx.unbind(0)
        logits, loss = self.nsp(idx, None, targets)
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
        # if the sequence context is growing too long we must crop it at block_size
        #idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
        # forward the model to get the logits for the index in the sequence
        seq_ids = idx.new_ones(idx.shape)
        logits, _ = self(torch.stack((idx, seq_ids)))
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
        #idx = torch.cat((idx, idx_next), dim=-1)

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
