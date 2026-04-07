"""
Activation Field Memory Transformer (AFMT).

Meme backbone transformer que le baseline, augmente du champ d'activation.
L'activation evolue a travers la profondeur du reseau :
- Blocs 0-1 -> Memory interaction 1 -> update activations
- Blocs 2-3 -> Memory interaction 2 -> update activations
- Blocs 4-5 -> Memory interaction 3 -> update activations

A chaque interaction, le foyer courant active ses voisins semantiques
via le graphe de propagation. Les couches profondes beneficient des
concepts actives par les couches precedentes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TransformerConfig, MemoryConfig
from src.transformer import TransformerBlock
from src.activation_field import ActivationField, FocusCrossAttention


class AFMTransformer(nn.Module):
    """Transformer augmente du champ d'activation a foyer mobile."""

    def __init__(self, t_config: TransformerConfig, m_config: MemoryConfig):
        super().__init__()
        self.t_config = t_config
        self.m_config = m_config

        # Embeddings (identiques au baseline)
        self.tok_emb = nn.Embedding(t_config.vocab_size, t_config.d_model)
        self.pos_emb = nn.Embedding(t_config.max_seq_len, t_config.d_model)
        self.drop = nn.Dropout(t_config.dropout)

        # Blocs transformer (identiques au baseline)
        self.blocks = nn.ModuleList([
            TransformerBlock(t_config) for _ in range(t_config.n_layers)
        ])

        # Champ d'activation
        self.activation_field = ActivationField(
            n_elements=m_config.n_elements,
            d_model=t_config.d_model,
            n_neighbors=m_config.n_neighbors,
            alpha=m_config.alpha,
            delta=m_config.delta,
            gamma=m_config.gamma,
            epsilon=m_config.epsilon,
        )

        # Cross-attention memoire avec foyer top-K
        # Adapter le nombre d'interactions au nombre de couches
        actual_mem_layers = min(m_config.n_memory_layers, t_config.n_layers)
        blocks_per_interaction = max(1, t_config.n_layers // actual_mem_layers)
        self.blocks_per_interaction = blocks_per_interaction
        self.memory_cross_attn = nn.ModuleList([
            FocusCrossAttention(
                d_model=t_config.d_model,
                n_heads=t_config.n_heads,
                focus_k=m_config.focus_k,
                dropout=t_config.dropout,
            )
            for _ in range(actual_mem_layers)
        ])

        # Tete de sortie
        self.ln_f = nn.LayerNorm(t_config.d_model)
        self.head = nn.Linear(t_config.d_model, t_config.vocab_size, bias=False)

        if t_config.weight_tying:
            self.head.weight = self.tok_emb.weight

        # Init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('fc2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * t_config.n_layers))

        # Init memoire depuis les token embeddings : brise la symetrie
        # Les elements memoire commencent avec du contenu semantique reel
        with torch.no_grad():
            n_mem = m_config.n_elements
            n_vocab = t_config.vocab_size
            if n_mem <= n_vocab:
                # Prendre les n_mem premiers tokens
                self.activation_field.memory.copy_(self.tok_emb.weight[:n_mem])
            else:
                # Repeter si plus d'elements que de tokens
                repeats = n_mem // n_vocab + 1
                expanded = self.tok_emb.weight.repeat(repeats, 1)[:n_mem]
                self.activation_field.memory.copy_(expanded)

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
        device = idx.device
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Embeddings
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Initialiser le champ d'activation
        activations = self.activation_field.get_resting_state(B, device)

        # Passer a travers les blocs avec interactions memoire
        mem_idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Interaction memoire apres chaque groupe de blocs
            if (i + 1) % self.blocks_per_interaction == 0 and mem_idx < len(self.memory_cross_attn):
                x, attn_weights = self.memory_cross_attn[mem_idx](
                    x, self.activation_field.memory, activations
                )
                activations = self.activation_field.update(activations, attn_weights)
                mem_idx += 1

        # Sortie
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, activations

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Detail par composant
        transformer_params = (
            sum(p.numel() for p in self.tok_emb.parameters())
            + sum(p.numel() for p in self.pos_emb.parameters())
            + sum(p.numel() for p in self.blocks.parameters())
            + sum(p.numel() for p in self.ln_f.parameters())
        )
        # head partage avec tok_emb si weight_tying
        if not self.t_config.weight_tying:
            transformer_params += sum(p.numel() for p in self.head.parameters())

        memory_params = sum(p.numel() for p in self.activation_field.parameters())
        cross_attn_params = sum(p.numel() for p in self.memory_cross_attn.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'transformer': transformer_params,
            'memory_elements': memory_params,
            'cross_attention': cross_attn_params,
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generation autoregressive avec evolution du champ d'activation."""
        device = idx.device
        activations = self.activation_field.get_resting_state(idx.shape[0], device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.t_config.max_seq_len:]
            B, T = idx_cond.shape
            pos = torch.arange(0, T, dtype=torch.long, device=device)

            x = self.drop(self.tok_emb(idx_cond) + self.pos_emb(pos))

            mem_idx = 0
            for i, block in enumerate(self.blocks):
                x = block(x)
                if (i + 1) % self.blocks_per_interaction == 0 and mem_idx < len(self.memory_cross_attn):
                    x, attn_weights = self.memory_cross_attn[mem_idx](
                        x, self.activation_field.memory, activations
                    )
                    activations = self.activation_field.update(activations, attn_weights)
                    mem_idx += 1

            x = self.ln_f(x)
            logits = self.head(x)[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
