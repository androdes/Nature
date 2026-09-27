"""
Memoire vivante : la cave et le caviste.

Trois idees, dans l'ordre (cf. memoire_vivante.md) :

1. La cave contient une information absente des poids : les cles/valeurs
   (K, V) des segments PASSES du document en cours. Elle vit en RAM CPU,
   jamais entierement en VRAM.

2. Le foyer est PAR TOKEN et CAUSAL : chaque token interroge la cave avec
   sa propre requete Q (celle de la self-attention, donc entrainee par
   gradient). La cave ne contient que des segments deja lus : le segment
   courant y est ecrit APRES la lecture. La memoire persiste d'un segment
   a l'autre et n'est videe qu'au changement de document.

3. Le champ d'activation devient le caviste : il ne pilote pas l'attention
   (le resultat est identique qu'il existe ou non). Il observe ce que le
   chef a vraiment consulte, propage l'activation aux fiches voisines dans
   le temps, et decide quelles fiches monter en VRAM pour le pas suivant.
   On mesure son taux de succes contre une politique LRU de meme capacite.
"""

import torch
import torch.nn.functional as F


class KNNMemory:
    """
    La cave : un anneau de max_size fiches (K, V) par flux, en RAM CPU.

    n_streams flux independants (un par ligne du batch). Chaque flux lit un
    document segment apres segment ; reset() vide les flux qui changent de
    document.
    """

    def __init__(self, n_streams, n_heads, head_dim, max_size,
                 storage_device='cpu', search_chunk=64):
        self.n_streams = n_streams
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_size = max_size
        self.device = torch.device(storage_device)
        self.search_chunk = search_chunk

        shape = (n_streams, n_heads, max_size, head_dim)
        self.keys = torch.zeros(shape, device=self.device)
        self.keys_n = torch.zeros(shape, device=self.device)  # normalisees, pour la recherche
        self.values = torch.zeros(shape, device=self.device)
        # Nombre total de fiches ecrites par flux depuis le dernier reset
        self.count = torch.zeros(n_streams, dtype=torch.long)

    def reset(self, mask=None):
        """Vide les flux indiques (mask: (B,) bool). None = tout vider."""
        if mask is None:
            self.count.zero_()
        else:
            self.count[mask.cpu()] = 0

    def sizes(self):
        """Nombre de fiches valides par flux."""
        return self.count.clamp(max=self.max_size)

    def is_empty(self):
        return bool((self.count == 0).all())

    @torch.no_grad()
    def write(self, k, v):
        """
        Range les fiches du segment courant. k, v: (B, H, T, D), detachees.
        Retourne les slots ecrits (B, T) pour le caviste.
        """
        B, H, T, D = k.shape
        k = k.detach().to(self.device, torch.float32)
        v = v.detach().to(self.device, torch.float32)
        if T > self.max_size:
            k, v = k[:, :, -self.max_size:], v[:, :, -self.max_size:]
            self.count += T - self.max_size
            T = self.max_size

        offsets = torch.arange(T)
        slots = (self.count.unsqueeze(1) + offsets.unsqueeze(0)) % self.max_size  # (B, T)
        idx = slots.to(self.device).view(B, 1, T, 1).expand(B, H, T, D)
        self.keys.scatter_(2, idx, k)
        self.keys_n.scatter_(2, idx, F.normalize(k, dim=-1))
        self.values.scatter_(2, idx, v)
        self.count += T
        return slots

    @torch.no_grad()
    def search(self, q, topk):
        """
        Chaque token cherche ses topk fiches les plus proches (cosinus).

        q: (B, H, T, D) sur n'importe quel device.
        Retourne, sur le device de q :
          k_sel, v_sel : (B, H, T, K, D) fiches brutes (sans gradient)
          valid        : (B, H, T, K) bool, False si le flux a moins de K fiches
          slots        : (B, H, T, K) long sur CPU, pour le caviste
        """
        B, H, T, D = q.shape
        K = min(topk, self.max_size)
        qn = F.normalize(q.detach().to(self.device, torch.float32), dim=-1)

        sizes = self.sizes().to(self.device)
        slot_ids = torch.arange(self.max_size, device=self.device)
        empty = (slot_ids.view(1, -1) >= sizes.view(-1, 1)).view(B, 1, 1, -1)  # (B,1,1,M)

        top_s, top_i = [], []
        for t0 in range(0, T, self.search_chunk):
            s = torch.einsum('bhtd,bhmd->bhtm', qn[:, :, t0:t0 + self.search_chunk], self.keys_n)
            s = s.masked_fill(empty, float('-inf'))
            vals, ids = s.topk(K, dim=-1)
            top_s.append(vals)
            top_i.append(ids)
        scores = torch.cat(top_s, dim=2)
        slots = torch.cat(top_i, dim=2)            # (B, H, T, K)
        valid = torch.isfinite(scores)

        flat = slots.reshape(B, H, T * K, 1).expand(B, H, T * K, D)
        k_sel = torch.gather(self.keys, 2, flat).view(B, H, T, K, D)
        v_sel = torch.gather(self.values, 2, flat).view(B, H, T, K, D)

        dev = q.device
        return (k_sel.to(dev, q.dtype), v_sel.to(dev, q.dtype),
                valid.to(dev), slots.cpu())


class Caviste:
    """
    Le champ d'activation, dans le role modeste : ranger la cave.

    Pour chaque flux, une activation par slot. A chaque pas :
      - on mesure combien des fiches consultees etaient deja montees en VRAM
        (prediction faite au pas precedent) ;
      - l'activation decroit, monte pour les fiches consultees (ponderee par
        l'attention reellement recue), et se propage aux voisines temporelles
        (les tokens juste avant/apres dans le document) ;
      - les vram_slots fiches les plus actives sont designees pour le pas
        suivant.
    En parallele, une politique LRU de meme capacite sert de controle.

    Rien ici ne change le calcul du modele : seule la question
    "quelles fiches auraient deja ete en VRAM ?" est en jeu.
    """

    def __init__(self, n_streams, max_size, vram_slots,
                 delta=0.1, gamma=1.0, alpha=0.5, radius=2, fresh=0.0):
        self.B = n_streams
        self.M = max_size
        self.C = min(vram_slots, max_size)
        self.delta = delta      # decay
        self.gamma = gamma      # ce qui a ete consulte monte
        self.alpha = alpha      # propagation aux voisines
        self.radius = radius    # portee de la propagation (en tokens)
        self.fresh = fresh      # activation initiale d'une fiche neuve

        self.act = torch.zeros(self.B, self.M)
        self.hot = torch.zeros(self.B, self.M, dtype=torch.bool)
        self.last_used = torch.full((self.B, self.M), -1, dtype=torch.long)
        self.lru_hot = torch.zeros(self.B, self.M, dtype=torch.bool)
        self.step = 0
        self.reset_stats()

    def reset_stats(self):
        self.stats = {'picked': 0, 'hits': 0, 'lru_hits': 0}

    def reset(self, mask=None):
        rows = slice(None) if mask is None else mask.cpu()
        self.act[rows] = 0
        self.hot[rows] = False
        self.last_used[rows] = -1
        self.lru_hot[rows] = False

    @torch.no_grad()
    def observe(self, slots, weights, valid):
        """
        slots:   (B, H, T, K) long, fiches consultees
        weights: (B, H, T, K) attention recue par chaque fiche
        valid:   (B, H, T, K) bool
        """
        self.step += 1
        B = slots.shape[0]
        slots = slots.reshape(B, -1)
        w = (weights.float().cpu() * valid.float().cpu()).reshape(B, -1)
        v = valid.cpu().reshape(B, -1)

        usage = torch.zeros(B, self.M)
        usage.scatter_add_(1, slots, w)
        used = torch.zeros(B, self.M, dtype=torch.bool)
        used.scatter_(1, slots, v)  # slots consultes (uniques)

        # 1) Mesure : les fiches consultees etaient-elles deja en VRAM ?
        self.stats['picked'] += int(used.sum())
        self.stats['hits'] += int((used & self.hot).sum())
        self.stats['lru_hits'] += int((used & self.lru_hot).sum())

        # 2) Mise a jour de l'activation
        usage = usage / usage.sum(dim=1, keepdim=True).clamp(min=1e-8)
        spread = torch.zeros_like(usage)
        for r in range(1, self.radius + 1):
            spread += usage.roll(r, dims=1) + usage.roll(-r, dims=1)
        self.act = (1 - self.delta) * self.act + self.gamma * usage + self.alpha * spread

        # 3) Controle LRU
        self.last_used[used] = self.step

    @torch.no_grad()
    def on_write(self, slots, sizes):
        """Fiches neuves ecrites (slots (B, T)), puis choix du prochain lot VRAM."""
        B = slots.shape[0]
        self.act.scatter_(1, slots, torch.full(slots.shape, self.fresh))
        # Une fiche neuve est "utilisee maintenant" pour la LRU (recence)
        self.last_used.scatter_(1, slots, torch.full(slots.shape, self.step))
        self._choose(sizes)

    def _choose(self, sizes):
        valid = torch.arange(self.M).view(1, -1) < sizes.view(-1, 1)
        C = self.C
        act = self.act.masked_fill(~valid, float('-inf'))
        idx = act.topk(C, dim=1).indices
        self.hot = torch.zeros(self.B, self.M, dtype=torch.bool)
        self.hot.scatter_(1, idx, True)
        self.hot &= valid

        lu = self.last_used.float().masked_fill(~valid, float('-inf'))
        idx = lu.topk(C, dim=1).indices
        self.lru_hot = torch.zeros(self.B, self.M, dtype=torch.bool)
        self.lru_hot.scatter_(1, idx, True)
        self.lru_hot &= valid

    def summary(self):
        p = max(self.stats['picked'], 1)
        return {
            'vram_slots': self.C,
            'picked': self.stats['picked'],
            'caviste_hit_rate': self.stats['hits'] / p,
            'lru_hit_rate': self.stats['lru_hits'] / p,
        }


class MemoryState:
    """La cave + le caviste d'une couche memoire, pour un ensemble de flux."""

    def __init__(self, n_streams, n_heads, head_dim, max_size, vram_slots,
                 caviste_kwargs=None):
        self.memory = KNNMemory(n_streams, n_heads, head_dim, max_size)
        self.caviste = Caviste(n_streams, max_size, vram_slots, **(caviste_kwargs or {}))

    def reset(self, mask=None):
        self.memory.reset(mask)
        self.caviste.reset(mask)
