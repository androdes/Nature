"""
Donnees en flux, document par document.

Contrairement a src/data.py (qui concatene tout puis decoupe en sequences
independantes), un flux lit UN document dans l'ordre, segment apres
segment. C'est ce qui donne a la cave quelque chose a retenir : le debut
du document, sorti de la fenetre, mais encore utile plus loin.
"""

import os
import numpy as np
import torch

from src.config import DataConfig
from src.data import load_tokenizer, is_clean_line

# Categories de tokens pour l'evaluation
CAT_NAMES = {
    0: 'nouveau',     # trigramme jamais vu plus tot dans le document
    1: 'proche',      # derniere occurrence dans le segment courant (vue par l'attention locale)
    2: 'loin',        # derniere occurrence avant le segment, mais dans la cave
    3: 'hors_cave',   # derniere occurrence plus ancienne que la cave
}


def load_documents(config: DataConfig, max_doc_tokens=65536):
    """
    Un fichier = un document, nettoye comme dans src/data.py.
    Les tres longs fichiers sont coupes en morceaux de max_doc_tokens.
    Retourne (train_docs, val_docs, tokenizer) ; chaque doc est un np.array.
    La validation = la fin (val_ratio) de chaque fichier.
    """
    tokenizer = load_tokenizer(config.tokenizer_path)
    train_docs, val_docs = [], []
    for fname in sorted(os.listdir(config.data_dir)):
        if not fname.endswith('.txt') or fname in config.exclude_files:
            continue
        with open(os.path.join(config.data_dir, fname), 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f if is_clean_line(l)]
        if not lines:
            continue
        ids = np.array(tokenizer.encode('\n'.join(lines)).ids, dtype=np.int64)
        n_val = int(len(ids) * config.val_ratio)
        parts = [(ids[:len(ids) - n_val], train_docs), (ids[len(ids) - n_val:], val_docs)]
        for arr, dest in parts:
            for s in range(0, len(arr), max_doc_tokens):
                piece = arr[s:s + max_doc_tokens]
                if len(piece) > 2 * config.seq_len:
                    dest.append(piece)
    n_tr = sum(len(d) for d in train_docs)
    n_va = sum(len(d) for d in val_docs)
    print(f"Documents: {len(train_docs)} train ({n_tr:,} tokens), {len(val_docs)} val ({n_va:,} tokens)")
    return train_docs, val_docs, tokenizer


class DocumentStreams:
    """
    n_streams lecteurs. Chacun lit un document du debut a la fin, segment
    par segment, puis en tire un autre (probabilite proportionnelle a la
    longueur, pour que chaque token ait la meme chance d'etre vu).
    """

    def __init__(self, docs, n_streams, seq_len, seed=0):
        self.docs = docs
        self.B = n_streams
        self.T = seq_len
        self.rng = np.random.default_rng(seed)
        lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self.p = lengths / lengths.sum()
        self.cur = [None] * n_streams
        self.pos = [0] * n_streams

    def _new_doc(self, b):
        self.cur[b] = int(self.rng.choice(len(self.docs), p=self.p))
        self.pos[b] = 0

    def next_batch(self):
        T = self.T
        x = np.zeros((self.B, T), dtype=np.int64)
        y = np.zeros((self.B, T), dtype=np.int64)
        reset = np.zeros(self.B, dtype=bool)
        for b in range(self.B):
            if self.cur[b] is None or self.pos[b] + T + 1 > len(self.docs[self.cur[b]]):
                self._new_doc(b)
                reset[b] = True
            d, p = self.docs[self.cur[b]], self.pos[b]
            x[b], y[b] = d[p:p + T], d[p + 1:p + T + 1]
            self.pos[b] += T
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(reset)


def token_categories(doc, seq_len, mem_size):
    """
    Categorie de chaque cible doc[p] (p >= 1) selon la derniere occurrence
    du trigramme (doc[p-2], doc[p-1], doc[p]) plus tot dans le document.
    Le segment j contient les entrees doc[jT : jT+T] et les cibles
    doc[jT+1 : jT+T+1].
    """
    cats = np.zeros(len(doc), dtype=np.int64)
    last = {}
    for p in range(2, len(doc)):
        key = (int(doc[p - 2]), int(doc[p - 1]), int(doc[p]))
        seg_start = ((p - 1) // seq_len) * seq_len
        prev = last.get(key)
        if prev is None:
            cats[p] = 0
        elif prev >= seg_start:
            cats[p] = 1
        elif seg_start - prev <= mem_size:
            cats[p] = 2
        else:
            cats[p] = 3
        last[key] = p
    return cats


def build_eval_batches(docs, n_streams, seq_len, cats_fn=None, cats=None, max_segments=None):
    """
    Range les documents de validation dans n_streams flux (glouton, par
    longueur) et prepare la liste des pas (x, y, cat, reset).
    Les flux plus courts sont completes avec des cibles ignorees (-100).
    cats_fn(doc) -> categorie par token, ou cats = liste deja calculee.
    """
    T = seq_len
    streams = [[] for _ in range(n_streams)]
    load = [0] * n_streams
    for i in sorted(range(len(docs)), key=lambda i: -len(docs[i])):
        b = int(np.argmin(load))
        doc = docs[i]
        if cats is not None:
            doc_cats = cats[i]
        elif cats_fn is not None:
            doc_cats = cats_fn(doc)
        else:
            doc_cats = np.zeros(len(doc), dtype=np.int64)
        n_seg = (len(doc) - 1) // T
        for j in range(n_seg):
            s = j * T
            streams[b].append((doc[s:s + T], doc[s + 1:s + T + 1], doc_cats[s + 1:s + T + 1], j == 0))
        load[b] += n_seg

    n_steps = max(len(s) for s in streams)
    if max_segments is not None:
        n_steps = min(n_steps, max_segments)
    batches = []
    for t in range(n_steps):
        x = np.zeros((n_streams, T), dtype=np.int64)
        y = np.full((n_streams, T), -100, dtype=np.int64)
        c = np.full((n_streams, T), -1, dtype=np.int64)
        r = np.ones(n_streams, dtype=bool)
        for b in range(n_streams):
            if t < len(streams[b]):
                xs, ys, cs, rs = streams[b][t]
                x[b], y[b], c[b], r[b] = xs, ys, cs, rs
        batches.append(tuple(torch.from_numpy(a) for a in (x, y, c, r)))
    return batches
