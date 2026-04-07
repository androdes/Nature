"""
Champ d'activation distribue avec foyer mobile et propagation apprise.

L'intuition : quand on pense "chat", "souris" et "jardin" s'activent
par propagation. Le foyer se deplace, mais rien ne meurt - les elements
quittes descendent en activation sans jamais atteindre zero.

Implementation :
- N elements memoire, chacun un vecteur appris de dimension d_model
- Activation continue dans [epsilon, 1] pour chaque element
- Graphe de propagation sparse appris (K voisins par element)
- Foyer determine par cross-attention entre hidden states et memoire
- L'activation module l'attention : les elements deja actifs sont plus accessibles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationField(nn.Module):
    """Champ d'activation distribue sur N elements memoire."""

    def __init__(self, n_elements, d_model, n_neighbors=32,
                 alpha=0.1, delta=0.05, gamma=0.5, epsilon=1e-6):
        super().__init__()
        self.n_elements = n_elements
        self.d_model = d_model
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

        # Contenu des elements memoire (appris)
        self.memory = nn.Parameter(torch.randn(n_elements, d_model) * 0.02)

        # Etat de repos : activation de base de chaque element (appris)
        self.resting_logits = nn.Parameter(torch.zeros(n_elements))

        # Graphe de propagation : indices des voisins (recalcules) + poids (appris)
        self.register_buffer(
            'neighbor_indices',
            torch.zeros(n_elements, n_neighbors, dtype=torch.long)
        )
        self.neighbor_weights = nn.Parameter(torch.zeros(n_elements, n_neighbors))

        self._init_graph_random()

    def _init_graph_random(self):
        """Initialise le graphe avec des voisins aleatoires."""
        for i in range(self.n_elements):
            candidates = torch.cat([
                torch.arange(0, i),
                torch.arange(i + 1, self.n_elements)
            ])
            perm = torch.randperm(len(candidates))[:self.n_neighbors]
            self.neighbor_indices[i] = candidates[perm]
        nn.init.uniform_(self.neighbor_weights, -0.1, 0.1)

    @torch.no_grad()
    def recompute_graph(self):
        """Recalcule les voisins par similarite cosinus des embeddings memoire."""
        mem_norm = F.normalize(self.memory.data, dim=1)
        # Calcul par blocs pour economiser la memoire
        block_size = 512
        for start in range(0, self.n_elements, block_size):
            end = min(start + block_size, self.n_elements)
            sim = mem_norm[start:end] @ mem_norm.T  # (block, N)
            # Exclure soi-meme
            for i in range(end - start):
                sim[i, start + i] = -float('inf')
            _, topk = sim.topk(self.n_neighbors, dim=1)
            self.neighbor_indices[start:end] = topk

    def get_resting_state(self, batch_size, device):
        """Activation initiale pour un batch."""
        # Sigmoid pour garantir (0, 1), jamais exactement 0
        resting = torch.sigmoid(self.resting_logits)
        resting = resting.clamp(min=self.epsilon)
        return resting.unsqueeze(0).expand(batch_size, -1).clone()

    def spread(self, activations):
        """Propage l'activation a travers le graphe de voisinage.

        Utilise softmax sur les poids voisins pour que la contribution
        totale soit une moyenne ponderee (bornee par max voisin),
        pas une somme qui explose avec K voisins.
        """
        # activations: (B, N)
        neighbor_acts = activations[:, self.neighbor_indices]  # (B, N, K)
        # Softmax : les poids par element somment a 1 -> moyenne ponderee
        weights = F.softmax(self.neighbor_weights, dim=-1)  # (N, K)
        # Moyenne ponderee des activations voisines
        spread_signal = (neighbor_acts * weights.unsqueeze(0)).sum(dim=-1)  # (B, N)
        return activations + self.alpha * spread_signal

    def update(self, activations, attention_weights):
        """
        Met a jour les activations apres une interaction memoire.

        Args:
            activations: (B, N) etat courant
            attention_weights: (B, N) poids d'attention moyens de la cross-attention
        Returns:
            activations mises a jour (B, N)
        """
        # Decay : tout descend lentement, rien ne meurt
        a = (1.0 - self.delta) * activations
        # Contribution du foyer : ce qui a ete regarde monte
        a = a + self.gamma * attention_weights
        # Propagation : les voisins des elements actifs s'activent
        a = self.spread(a)
        # Clamp : jamais zero, jamais au-dessus de 1
        a = a.clamp(self.epsilon, 1.0)
        return a

    def get_diagnostics(self, activations):
        """Diagnostics pour le logging."""
        with torch.no_grad():
            return {
                'act_mean': activations.mean().item(),
                'act_std': activations.std().item(),
                'act_max': activations.max().item(),
                'act_min': activations.min().item(),
                'act_top10_mean': activations.topk(10, dim=-1).values.mean().item(),
                'act_sparsity': (activations < 0.1).float().mean().item(),
            }


class FocusCrossAttention(nn.Module):
    """
    Cross-attention avec foyer mobile : seuls les top-K elements les plus
    actifs participent a l'attention. Softmax sur K=64 au lieu de N=2048,
    ce qui donne une attention 32x plus focalisee.

    C'est le "VRAM window" de la vision originale : a chaque instant,
    le foyer ne voit qu'un petit sous-ensemble. Le reste existe en
    latence dans le champ d'activation.
    """

    def __init__(self, d_model, n_heads=4, focus_k=64, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model
        self.focus_k = focus_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

        # Projection pour la selection du foyer (content-based)
        self.focus_proj = nn.Linear(d_model, d_model)
        # Poids relatif activation vs pertinence dans la selection
        self.activation_weight = nn.Parameter(torch.tensor(1.0))

        # Gate appris : commence a 0.5
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, h, memory, activations):
        """
        Args:
            h: (B, T, d_model) hidden states du transformer
            memory: (N, d_model) elements memoire
            activations: (B, N) niveaux d'activation courants

        Returns:
            h_updated: (B, T, d_model)
            full_attn: (B, N) attention projetee sur tout N (0 pour non-selectionnes)
        """
        B, T, d = h.shape
        N = memory.shape[0]
        K = min(self.focus_k, N)
        H = self.n_heads
        D = self.head_dim

        # Pre-norm (utilise aussi pour la selection du foyer)
        h_norm = self.ln(h)

        # Selection du foyer : pertinence au contenu + activation
        h_summary = h_norm.mean(dim=1)  # (B, d) - resume de la sequence
        focus_query = self.focus_proj(h_summary)  # (B, d)
        relevance = focus_query @ memory.T  # (B, N) - pertinence semantique
        # Score = pertinence + inertie d'activation
        selection = relevance + self.activation_weight * activations
        topk_vals, topk_idx = selection.topk(K, dim=-1)  # (B, K)

        # Rassembler les embeddings memoire du foyer
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, d)  # (B, K, d)
        topk_mem = memory.unsqueeze(0).expand(B, -1, -1).gather(1, idx_expanded)  # (B, K, d)

        # Projections
        q = self.q_proj(h_norm).view(B, T, H, D).transpose(1, 2)       # (B, H, T, D)
        k = self.k_proj(topk_mem).view(B, K, H, D).permute(0, 2, 1, 3)  # (B, H, K, D)
        v = self.v_proj(topk_mem).view(B, K, H, D).permute(0, 2, 1, 3)  # (B, H, K, D)

        # Scores d'attention sur le foyer uniquement
        scores = torch.einsum('bhtd,bhkd->bhtk', q, k) / (D ** 0.5)  # (B, H, T, K)

        attn = F.softmax(scores, dim=-1)  # (B, H, T, K) - sharp car K petit
        attn = self.attn_drop(attn)

        # Lecture memoire
        out = torch.einsum('bhtk,bhkd->bhtd', attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        out = self.resid_drop(self.out_proj(out))

        # Injection gatee
        gate = torch.sigmoid(self.gate_logit)
        h_updated = h + gate * out

        # Projeter l'attention du foyer sur tout N pour l'update d'activation
        mean_attn_topk = attn.mean(dim=(1, 2))  # (B, K)
        full_attn = torch.zeros(B, N, device=h.device)
        full_attn.scatter_(1, topk_idx, mean_attn_topk)

        return h_updated, full_attn
