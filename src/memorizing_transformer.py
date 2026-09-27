"""
Transformer a memoire vivante (style Memorizing Transformer, Wu et al. 2022).

Un GPT causal standard (blocs de src/transformer.py) dont UNE couche est
remplacee par un bloc qui, en plus de la self-attention locale sur le
segment courant, consulte la cave (src/knn_memory.py) :

  y = g * attention(q, fiches_recuperees) + (1 - g) * attention_locale(q)

avec une porte g apprise par tete. La requete q est celle de la
self-attention : elle recoit donc du gradient (contrairement a la requete
de retrieval de hierarchical_memory.py, detachee avant FAISS).

Le modele traite un document segment par segment. Les fiches du segment
courant sont rangees dans la cave APRES la lecture : un token ne peut
consulter que le passe. use_memory=False donne exactement le meme modele,
sans cave (controle a parametres identiques).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TransformerConfig
from src.transformer import TransformerBlock, FeedForward
from src.knn_memory import MemoryState


class KNNAugmentedBlock(nn.Module):
    """Bloc transformer dont l'attention lit aussi la cave."""

    def __init__(self, config: TransformerConfig, topk=32):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.topk = topk
        self.dropout = config.dropout

        self.ln1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.resid_drop = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

        # Porte par tete : sigmoid(0) = 0.5 au depart
        self.gate_logit = nn.Parameter(torch.zeros(config.n_heads))
        self.last_gate = None

    def _attend(self, x, state):
        B, T, C = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, D)

        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)

        if state is not None and not state.memory.is_empty():
            k_mem, v_mem, valid, slots = state.memory.search(q, self.topk)
            s = torch.einsum('bhtd,bhtkd->bhtk', q, k_mem) / math.sqrt(D)
            s = s.masked_fill(~valid, float('-inf'))
            has_mem = valid.any(dim=-1, keepdim=True)          # (B, H, T, 1)
            w = F.softmax(s.masked_fill(~has_mem, 0.0), dim=-1)
            w = w * valid                                     # zero si aucune fiche
            y_mem = torch.einsum('bhtk,bhtkd->bhtd', w, v_mem)

            g = torch.sigmoid(self.gate_logit).view(1, H, 1, 1) * has_mem
            y = g * y_mem + (1 - g) * y
            state.caviste.observe(slots, w.detach(), valid)
            self.last_gate = torch.sigmoid(self.gate_logit).detach()

        if state is not None:
            written = state.memory.write(k, v)   # rangees APRES la lecture
            state.caviste.on_write(written, state.memory.sizes())

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))

    def forward(self, x, state=None):
        x = x + self._attend(self.ln1(x), state)
        x = x + self.ff(self.ln2(x))
        return x


class MemorizingTransformer(nn.Module):
    """GPT causal lisant un document segment par segment, avec une cave."""

    def __init__(self, config: TransformerConfig, mem_layer=None, topk=32,
                 use_memory=True):
        super().__init__()
        self.config = config
        self.use_memory = use_memory
        self.mem_layer = config.n_layers // 2 if mem_layer is None else mem_layer
        self.topk = topk

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            KNNAugmentedBlock(config, topk) if i == self.mem_layer else TransformerBlock(config)
            for i in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('fc2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

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

    def new_state(self, n_streams, mem_size, vram_slots, caviste_kwargs=None):
        """Une cave vide pour n_streams flux (None si use_memory=False)."""
        if not self.use_memory:
            return None
        c = self.config
        return MemoryState(n_streams, c.n_heads, c.d_model // c.n_heads,
                           mem_size, vram_slots, caviste_kwargs)

    def forward(self, idx, targets=None, state=None, reset=None):
        """
        idx: (B, T) segment courant de chaque flux.
        reset: (B,) bool, True si ce segment commence un nouveau document.
        state: MemoryState (ou None) ; mis a jour en place.
        """
        B, T = idx.shape
        if state is not None and reset is not None:
            state.reset(reset)

        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, state) if i == self.mem_layer else block(x)

        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-100)
        return logits, loss

    def gate(self):
        return torch.sigmoid(self.blocks[self.mem_layer].gate_logit).detach()

    def count_parameters(self):
        return {'total': sum(p.numel() for p in self.parameters())}
