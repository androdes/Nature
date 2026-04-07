"""
Memoire externe a grande echelle via retrieval FAISS.

Le transformer a appris a consulter une memoire de 512 elements.
Ici on lui donne une memoire de N elements (N potentiellement 50K+)
mais a chaque forward pass, seuls K elements pertinents sont recuperes
via FAISS et passes en cross-attention.

Cout VRAM : O(K) quel que soit N.
Cout disque/RAM : O(N) pour l'index FAISS + les vecteurs.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from src.config import TransformerConfig
from src.transformer import TransformerBlock


class FAISSMemoryBank:
    """Banque de memoire indexee par FAISS. Vit en RAM/disque."""

    def __init__(self, d_model, n_elements=None):
        self.d_model = d_model
        self.index = faiss.IndexFlatIP(d_model)  # Inner product (cosine apres normalisation)
        self.vectors = None  # (N, d) numpy array

    def build(self, vectors: np.ndarray):
        """Construit l'index depuis un array de vecteurs."""
        # Normaliser pour cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = vectors / norms
        self.vectors = vectors.copy()
        self.index.reset()
        self.index.add(normalized.astype(np.float32))

    def search(self, queries: np.ndarray, k: int):
        """
        Recherche les K plus proches voisins.
        queries: (B, d) numpy array
        Retourne: indices (B, K), distances (B, K)
        """
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = queries / norms
        distances, indices = self.index.search(normalized.astype(np.float32), k)
        return indices, distances

    @property
    def n_elements(self):
        return self.index.ntotal


class RetrievalCrossAttention(nn.Module):
    """Cross-attention sur les K elements recuperes par FAISS."""

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

        # Projection pour la requete de retrieval
        self.retrieval_proj = nn.Linear(d_model, d_model)

        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, h, retrieved_memory):
        """
        h: (B, T, d) hidden states
        retrieved_memory: (B, K, d) elements recuperes par FAISS
        """
        B, T, d = h.shape
        K = retrieved_memory.shape[1]
        H, D = self.n_heads, self.head_dim

        h_norm = self.ln(h)
        q = self.q_proj(h_norm).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(retrieved_memory).view(B, K, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(retrieved_memory).view(B, K, H, D).permute(0, 2, 1, 3)

        scores = torch.einsum('bhtd,bhkd->bhtk', q, k) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhtk,bhkd->bhtd', attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        out = self.resid_drop(self.out_proj(out))

        gate = torch.sigmoid(self.gate_logit)
        return h + gate * out

    def compute_retrieval_query(self, h):
        """Produit la requete pour FAISS a partir des hidden states."""
        h_norm = self.ln(h)
        h_summary = h_norm.mean(dim=1)  # (B, d)
        return self.retrieval_proj(h_summary)  # (B, d)


class RetrievalMemoryTransformer(nn.Module):
    """
    Transformer avec memoire externe retrievable.
    La memoire peut contenir N elements (N arbitrairement grand).
    A chaque forward pass, K elements sont recuperes et consultes.
    """

    def __init__(self, t_config: TransformerConfig, retrieve_k: int = 64,
                 mem_layer_position: int = None):
        super().__init__()
        self.t_config = t_config
        self.retrieve_k = retrieve_k

        # Position de la cross-attention (par defaut : apres la moitie des blocs)
        if mem_layer_position is None:
            mem_layer_position = t_config.n_layers // 2
        self.mem_layer_position = mem_layer_position

        self.tok_emb = nn.Embedding(t_config.vocab_size, t_config.d_model)
        self.pos_emb = nn.Embedding(t_config.max_seq_len, t_config.d_model)
        self.drop = nn.Dropout(t_config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(t_config) for _ in range(t_config.n_layers)
        ])

        self.retrieval_cross_attn = RetrievalCrossAttention(
            t_config.d_model, t_config.n_heads, t_config.dropout
        )

        self.ln_f = nn.LayerNorm(t_config.d_model)
        self.head = nn.Linear(t_config.d_model, t_config.vocab_size, bias=False)
        if t_config.weight_tying:
            self.head.weight = self.tok_emb.weight

        # Memory bank (CPU/disque via FAISS)
        self.memory_bank = FAISSMemoryBank(t_config.d_model)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('fc2.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * t_config.n_layers))

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

    def build_memory_from_embeddings(self, n_elements=None):
        """Construit la memoire depuis les token embeddings (parametrique)."""
        with torch.no_grad():
            emb = self.tok_emb.weight.detach().cpu().numpy()
            if n_elements is not None and n_elements > len(emb):
                # Repeter + bruit pour diversifier
                repeats = n_elements // len(emb) + 1
                expanded = np.tile(emb, (repeats, 1))[:n_elements]
                noise = np.random.randn(*expanded.shape).astype(np.float32) * 0.01
                expanded = expanded + noise
                self.memory_bank.build(expanded)
            else:
                n = n_elements or len(emb)
                self.memory_bank.build(emb[:n])

    def build_memory_from_data(self, vectors: np.ndarray):
        """Construit la memoire depuis des vecteurs externes (non-parametrique)."""
        self.memory_bank.build(vectors)

    def _retrieve(self, h, device):
        """Recupere K elements pertinents depuis la memoire FAISS."""
        # Produire la requete de retrieval
        query = self.retrieval_cross_attn.compute_retrieval_query(h)  # (B, d)
        query_np = query.detach().cpu().numpy()

        # Recherche FAISS (CPU)
        indices, _ = self.memory_bank.search(query_np, self.retrieve_k)

        # Recuperer les vecteurs correspondants
        retrieved = self.memory_bank.vectors[indices]  # (B, K, d)
        return torch.tensor(retrieved, dtype=torch.float32, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(T, dtype=torch.long, device=device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i == self.mem_layer_position and self.memory_bank.n_elements > 0:
                retrieved = self._retrieve(x, device)
                x = self.retrieval_cross_attn(x, retrieved)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {
            'total_trainable': total,
            'memory_bank_size': self.memory_bank.n_elements,
            'memory_bank_bytes': self.memory_bank.n_elements * self.t_config.d_model * 4 if self.memory_bank.vectors is not None else 0,
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
