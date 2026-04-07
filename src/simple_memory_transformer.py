"""
Ablation : Simple Memory Transformer.
Cross-attention a une memoire externe apprise, SANS champ d'activation.
Pas de spreading, pas de decay, pas de foyer mobile.
Juste : "est-ce que la memoire externe aide du tout?"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TransformerConfig
from src.transformer import TransformerBlock


class SimpleMemoryCrossAttention(nn.Module):
    """Cross-attention plain a une memoire apprise. Pas de routing, pas d'activation."""

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, h, memory):
        B, T, d = h.shape
        N = memory.shape[0]
        H, D = self.n_heads, self.head_dim

        h_norm = self.ln(h)
        q = self.q_proj(h_norm).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(memory).view(N, H, D).permute(1, 0, 2)
        v = self.v_proj(memory).view(N, H, D).permute(1, 0, 2)

        scores = torch.einsum('bhtd,hnd->bhtn', q, k) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhtn,hnd->bhtd', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        out = self.resid_drop(self.out_proj(out))

        gate = torch.sigmoid(self.gate_logit)
        return h + gate * out


class SimpleMemoryTransformer(nn.Module):
    """Transformer + memoire externe sans activation field."""

    def __init__(self, t_config: TransformerConfig, n_mem_elements: int = 256,
                 n_mem_layers: int = 1):
        super().__init__()
        self.t_config = t_config

        self.tok_emb = nn.Embedding(t_config.vocab_size, t_config.d_model)
        self.pos_emb = nn.Embedding(t_config.max_seq_len, t_config.d_model)
        self.drop = nn.Dropout(t_config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(t_config) for _ in range(t_config.n_layers)
        ])

        # Memoire externe apprise
        self.memory = nn.Parameter(torch.randn(n_mem_elements, t_config.d_model) * 0.02)

        # Cross-attention(s)
        self.n_mem_layers = n_mem_layers
        actual_layers = min(n_mem_layers, t_config.n_layers)
        self.blocks_per_mem = max(1, t_config.n_layers // actual_layers)
        self.mem_cross_attn = nn.ModuleList([
            SimpleMemoryCrossAttention(t_config.d_model, t_config.n_heads, t_config.dropout)
            for _ in range(actual_layers)
        ])

        self.ln_f = nn.LayerNorm(t_config.d_model)
        self.head = nn.Linear(t_config.d_model, t_config.vocab_size, bias=False)
        if t_config.weight_tying:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('fc2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * t_config.n_layers))

        # Init memoire depuis token embeddings
        with torch.no_grad():
            n = min(n_mem_elements, t_config.vocab_size)
            self.memory[:n].copy_(self.tok_emb.weight[:n])

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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        mem_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) % self.blocks_per_mem == 0 and mem_idx < len(self.mem_cross_attn):
                x = self.mem_cross_attn[mem_idx](x, self.memory)
                mem_idx += 1

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'memory': self.memory.numel(),
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.t_config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
