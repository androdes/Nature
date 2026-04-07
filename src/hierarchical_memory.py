"""
Memoire hierarchique VRAM/RAM/Disque pour transformer.

La memoire peut contenir N elements (N >> capacite VRAM).
A chaque batch, K elements pertinents sont charges en VRAM via FAISS.
Les gradients sont accumules et appliques periodiquement.

La VRAM ne voit jamais toute la memoire. Mais tout est accessible.
"""

import math
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from collections import OrderedDict

from src.config import TransformerConfig
from src.transformer import TransformerBlock


class HierarchicalMemoryBank:
    """
    Memoire repartie sur disque (numpy memmap) + RAM (cache LRU) + VRAM (a la demande).
    Index FAISS pour retrieval rapide depuis n'importe quel niveau.
    """

    def __init__(self, n_elements, d_model, ram_cache_size=10000,
                 storage_path=None):
        self.n_elements = n_elements
        self.d_model = d_model
        self.ram_cache_size = ram_cache_size

        # Disque : memmap numpy
        if storage_path is None:
            storage_path = "D:/Nature/runs/membank.dat"
        self.storage_path = storage_path
        self.vectors = np.memmap(
            storage_path, dtype=np.float32, mode='w+',
            shape=(n_elements, d_model)
        )

        # RAM : cache LRU des elements recemment accedes
        self.ram_cache = OrderedDict()  # idx -> numpy array

        # FAISS index (vit en RAM)
        self.index = faiss.IndexFlatIP(d_model)

        # Stats
        self.stats = {
            'vram_hits': 0, 'ram_hits': 0, 'disk_hits': 0,
            'total_retrievals': 0, 'retrieval_time_ms': 0,
            'unique_elements_accessed': set(),
        }

    def initialize(self, init_vectors: np.ndarray):
        """Initialise la memoire depuis un array."""
        assert init_vectors.shape == (self.n_elements, self.d_model)
        self.vectors[:] = init_vectors
        self.vectors.flush()
        self._rebuild_index()

    def _rebuild_index(self):
        """Reconstruit l'index FAISS depuis les vecteurs sur disque."""
        self.index.reset()
        # Charger par blocs pour eviter de tout mettre en RAM
        block_size = 10000
        for start in range(0, self.n_elements, block_size):
            end = min(start + block_size, self.n_elements)
            block = np.array(self.vectors[start:end])  # copie depuis memmap
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.index.add((block / norms).astype(np.float32))

    def retrieve(self, queries: np.ndarray, k: int):
        """
        Recupere les K plus proches voisins.
        queries: (B, d) numpy
        Retourne: indices (B, K), vectors (B, K, d)
        """
        t0 = time.time()

        # Normaliser queries
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        _, indices = self.index.search((queries / norms).astype(np.float32), k)

        # Charger les vecteurs depuis le niveau le plus rapide disponible
        B, K = indices.shape
        result = np.zeros((B, K, self.d_model), dtype=np.float32)

        for b in range(B):
            for ki in range(K):
                idx = int(indices[b, ki])
                if idx < 0:  # FAISS peut retourner -1
                    continue

                self.stats['unique_elements_accessed'].add(idx)

                if idx in self.ram_cache:
                    result[b, ki] = self.ram_cache[idx]
                    self.ram_cache.move_to_end(idx)
                    self.stats['ram_hits'] += 1
                else:
                    result[b, ki] = self.vectors[idx]
                    self.stats['disk_hits'] += 1
                    # Ajouter au cache RAM
                    self._cache_put(idx, result[b, ki].copy())

        dt = (time.time() - t0) * 1000
        self.stats['total_retrievals'] += 1
        self.stats['retrieval_time_ms'] += dt

        return indices, result

    def _cache_put(self, idx, vector):
        """Ajoute un element au cache RAM avec eviction LRU."""
        if idx in self.ram_cache:
            self.ram_cache.move_to_end(idx)
            return
        if len(self.ram_cache) >= self.ram_cache_size:
            self.ram_cache.popitem(last=False)  # evicte le plus ancien
        self.ram_cache[idx] = vector

    def update_vectors(self, indices, new_vectors):
        """
        Met a jour les vecteurs aux indices donnes (apres gradient step).
        indices: list d'ints
        new_vectors: numpy (len, d)
        """
        for i, idx in enumerate(indices):
            self.vectors[idx] = new_vectors[i]
            if idx in self.ram_cache:
                self.ram_cache[idx] = new_vectors[i].copy()
        self.vectors.flush()

    def get_stats(self):
        total = self.stats['ram_hits'] + self.stats['disk_hits']
        return {
            'total_elements': self.n_elements,
            'ram_cache_size': len(self.ram_cache),
            'ram_cache_max': self.ram_cache_size,
            'total_accesses': total,
            'ram_hit_rate': self.stats['ram_hits'] / max(total, 1),
            'disk_hit_rate': self.stats['disk_hits'] / max(total, 1),
            'avg_retrieval_ms': self.stats['retrieval_time_ms'] / max(self.stats['total_retrievals'], 1),
            'unique_accessed': len(self.stats['unique_elements_accessed']),
            'coverage': len(self.stats['unique_elements_accessed']) / self.n_elements,
        }


class HierarchicalCrossAttention(nn.Module):
    """Cross-attention sur K elements recuperes depuis la hierarchie."""

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

        self.retrieval_proj = nn.Linear(d_model, d_model)
        self.gate_logit = nn.Parameter(torch.tensor(0.0))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def compute_retrieval_query(self, h):
        """Produit la requete pour le retrieval FAISS."""
        h_norm = self.ln(h)
        return self.retrieval_proj(h_norm.mean(dim=1))  # (B, d)

    def forward(self, h, retrieved_memory):
        """
        h: (B, T, d)
        retrieved_memory: (B, K, d) — deja en VRAM
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


class HierarchicalMemoryTransformer(nn.Module):
    """
    Transformer avec memoire hierarchique VRAM/RAM/Disque.

    La memoire contient N elements appris, mais seuls K sont en VRAM
    a chaque batch. Le reste vit en RAM (cache) ou sur disque (memmap).
    """

    def __init__(self, t_config: TransformerConfig, n_mem_elements=50000,
                 retrieve_k=64, ram_cache_size=10000, storage_path=None):
        super().__init__()
        self.t_config = t_config
        self.retrieve_k = retrieve_k
        self.n_mem_elements = n_mem_elements

        # Transformer standard
        self.tok_emb = nn.Embedding(t_config.vocab_size, t_config.d_model)
        self.pos_emb = nn.Embedding(t_config.max_seq_len, t_config.d_model)
        self.drop = nn.Dropout(t_config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(t_config) for _ in range(t_config.n_layers)
        ])

        # Cross-attention hierarchique (1 layer au milieu)
        self.mem_layer_position = t_config.n_layers // 2
        self.cross_attn = HierarchicalCrossAttention(
            t_config.d_model, t_config.n_heads, t_config.dropout
        )

        self.ln_f = nn.LayerNorm(t_config.d_model)
        self.head = nn.Linear(t_config.d_model, t_config.vocab_size, bias=False)
        if t_config.weight_tying:
            self.head.weight = self.tok_emb.weight

        # Memoire hierarchique (hors parametres PyTorch — vit sur disque/RAM)
        self.memory_bank = HierarchicalMemoryBank(
            n_mem_elements, t_config.d_model,
            ram_cache_size=ram_cache_size,
            storage_path=storage_path,
        )

        # Buffer pour accumuler les gradients des elements memoire
        self._retrieved_indices = None  # (B, K) indices du dernier retrieval
        self._retrieved_vram = None     # (B, K, d) tenseurs en VRAM avec grad

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

    def initialize_memory(self):
        """Initialise la memoire depuis les token embeddings + bruit."""
        with torch.no_grad():
            emb = self.tok_emb.weight.detach().cpu().numpy()
            n_vocab = len(emb)
            repeats = self.n_mem_elements // n_vocab + 1
            expanded = np.tile(emb, (repeats, 1))[:self.n_mem_elements]
            noise = np.random.randn(*expanded.shape).astype(np.float32) * 0.01
            self.memory_bank.initialize(expanded + noise)

    def _retrieve_memory(self, h, device):
        """Retrieve K elements depuis la hierarchie."""
        query = self.cross_attn.compute_retrieval_query(h)
        query_np = query.detach().cpu().numpy()

        indices, vectors = self.memory_bank.retrieve(query_np, self.retrieve_k)

        # Mettre en VRAM comme tenseur avec gradient
        retrieved = torch.tensor(vectors, dtype=torch.float32,
                                 device=device, requires_grad=True)

        self._retrieved_indices = indices
        self._retrieved_vram = retrieved

        return retrieved

    def apply_memory_gradients(self, lr=1e-3):
        """
        Applique les gradients accumules aux elements memoire sur disque.
        Appele APRES loss.backward().
        """
        if self._retrieved_vram is None or self._retrieved_vram.grad is None:
            return 0

        grad = self._retrieved_vram.grad.detach().cpu().numpy()
        indices = self._retrieved_indices
        B, K = indices.shape

        n_updated = 0
        for b in range(B):
            for ki in range(K):
                idx = int(indices[b, ki])
                if idx < 0:
                    continue
                # SGD simple sur l'element memoire
                current = self.memory_bank.vectors[idx].copy()
                updated = current - lr * grad[b, ki]
                self.memory_bank.vectors[idx] = updated
                if idx in self.memory_bank.ram_cache:
                    self.memory_bank.ram_cache[idx] = updated.copy()
                n_updated += 1

        self.memory_bank.vectors.flush()
        return n_updated

    def rebuild_index(self):
        """Reconstruit l'index FAISS (appeler periodiquement)."""
        self.memory_bank._rebuild_index()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(T, dtype=torch.long, device=device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i == self.mem_layer_position:
                retrieved = self._retrieve_memory(x, device)
                x = self.cross_attn(x, retrieved)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters())
        return {
            'trainable_vram': trainable,
            'memory_elements': self.n_mem_elements,
            'memory_params': self.n_mem_elements * self.t_config.d_model,
            'memory_bytes': self.n_mem_elements * self.t_config.d_model * 4,
            'total_params': trainable + self.n_mem_elements * self.t_config.d_model,
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
