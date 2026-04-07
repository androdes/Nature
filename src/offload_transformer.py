"""
Transformer avec optimizer offloade.

Strategie simple et robuste :
- Forward + backward : TOUT en VRAM (pas de rotation pendant le compute)
- Optimizer step : fait sur CPU, bloc par bloc, pour eviter l'OOM
  des states Adam (momentum + variance = 2x la taille des poids)

Pour les modeles qui ne tiennent meme pas en VRAM pour le compute :
- Gradient checkpointing reduit la VRAM des activations
- Si les POIDS ne tiennent pas, on reduit la taille du modele

Cette approche maximise l'utilisation GPU (pas de dents de scie)
et garde la simplicite (pas de hooks, pas de rotation complexe).
"""

import math
import time
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.config import TransformerConfig
from src.transformer import TransformerBlock


class OffloadTransformer(nn.Module):

    def __init__(self, config: TransformerConfig, use_checkpointing=True):
        super().__init__()
        self.config = config
        self.use_checkpointing = use_checkpointing

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('fc2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        self._stats = {'compute_ms': 0, 'optim_ms': 0, 'n_steps': 0}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def to_device(self, device):
        """Met tout le modele en VRAM pour le compute."""
        self.to(device)
        return self

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        t0 = time.time()
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        self._stats['compute_ms'] += (time.time() - t0) * 1000

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def setup_block_optimizers(self, lr=3e-4, weight_decay=0.01):
        """
        Cree un optimizer par bloc + un pour les embeddings.
        Les block optimizers operent sur GPU. Leurs Adam states
        sont deplacees CPU <-> GPU bloc par bloc pendant le step.
        """
        self._gpu_device = next(self.parameters()).device
        self._block_optimizers = []

        for block in self.blocks:
            opt = torch.optim.AdamW(
                block.parameters(), lr=lr,
                weight_decay=weight_decay, betas=(0.9, 0.95),
                foreach=False,  # foreach incompatible avec state migration CPU/GPU
            )
            self._block_optimizers.append(opt)

        # Embeddings : optimizer GPU direct (petit, states restent en VRAM)
        emb_params = list(self.tok_emb.parameters()) + list(self.pos_emb.parameters()) + \
                     list(self.ln_f.parameters())
        self._emb_optimizer = torch.optim.AdamW(
            emb_params, lr=lr,
            weight_decay=weight_decay, betas=(0.9, 0.95),
        )

    def cpu_optimizer_step(self, optimizer=None):
        """
        Block-by-block GPU optimizer :
        Pour chaque bloc, charge ses Adam states en VRAM, step sur GPU, offload.
        Le GPU fait le calcul Adam, le CPU ne fait que du stockage.
        """
        t0 = time.time()
        device = self._gpu_device

        # Embeddings : step direct (states deja en VRAM)
        self._emb_optimizer.step()
        self._emb_optimizer.zero_grad(set_to_none=True)

        # Blocs : un par un, states CPU <-> GPU
        for block, opt in zip(self.blocks, self._block_optimizers):
            # Charger Adam states -> VRAM
            for p in block.parameters():
                if p in opt.state and opt.state[p]:
                    for k, v in opt.state[p].items():
                        if isinstance(v, torch.Tensor) and v.device.type == 'cpu':
                            opt.state[p][k] = v.to(device, non_blocking=True)

            # Pas de sync : les ops GPU sont ordonnees dans le meme stream
            opt.step()
            opt.zero_grad(set_to_none=True)

            # Offload Adam states -> CPU
            for p in block.parameters():
                if p in opt.state and opt.state[p]:
                    for k, v in opt.state[p].items():
                        if isinstance(v, torch.Tensor) and v.device.type == 'cuda':
                            opt.state[p][k] = v.to('cpu', non_blocking=True)

        torch.cuda.synchronize()
        self._stats['optim_ms'] += (time.time() - t0) * 1000
        self._stats['n_steps'] += 1

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        return {
            'total': total,
            'per_block': per_block,
            'n_blocks': len(self.blocks),
            'weight_mb': total * 4 / 1024 / 1024,
        }

    def get_stats(self):
        n = max(self._stats['n_steps'], 1)
        return {
            'avg_compute_ms': self._stats['compute_ms'] / n,
            'avg_optim_ms': self._stats['optim_ms'] / n,
        }

    def reset_stats(self):
        self._stats = {'compute_ms': 0, 'optim_ms': 0, 'n_steps': 0}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
