"""Exp 9 : Memoire de corpus — retrieval de chunks reels.

Etapes :
1. Charger le baseline 6L pre-entraine (best.pt)
2. Encoder tout le corpus en hidden states (couche milieu)
3. Chunker en vecteurs de 32 tokens (mean pool) -> ~80K chunks
4. Construire un index FAISS
5. Entrainer un RetrievalMemoryTransformer qui consulte ces chunks reels

Hypothese : si le contenu de la memoire est du VRAI contexte semantique
(pas des embeddings appris), le gain devrait depasser le plafond de 1.3%.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.transformer import BaselineTransformer
from src.retrieval_memory import RetrievalMemoryTransformer
from src.train import train


def encode_corpus(model, train_ds, device, chunk_size=32):
    """Encode le corpus en hidden states a la couche milieu."""
    model.eval()
    all_vectors = []
    loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)

    mid_layer = len(model.blocks) // 2  # couche 3 pour un 6L

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            B, T = x.shape

            # Forward partiel jusqu'a la couche milieu
            pos = torch.arange(T, dtype=torch.long, device=device)
            h = model.drop(model.tok_emb(x) + model.pos_emb(pos))
            for i in range(mid_layer + 1):
                h = model.blocks[i](h)
            # h: (B, T, d_model)

            # Chunker : mean pool par groupes de chunk_size tokens
            n_chunks = T // chunk_size
            for c in range(n_chunks):
                chunk = h[:, c*chunk_size:(c+1)*chunk_size, :]  # (B, chunk_size, d)
                vec = chunk.mean(dim=1)  # (B, d)
                all_vectors.append(vec.cpu().numpy())

    vectors = np.concatenate(all_vectors, axis=0)  # (N_chunks, d)
    print(f"Corpus encode: {vectors.shape[0]:,} chunks de {chunk_size} tokens, dim={vectors.shape[1]}")
    return vectors


def run_one(n_chunks, corpus_vectors, seed, train_ds, val_ds, tokenizer, t_cfg):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = RetrievalMemoryTransformer(t_cfg, retrieve_k=64)

    # Utiliser les N premiers chunks (ou tous)
    n = min(n_chunks, len(corpus_vectors)) if n_chunks else len(corpus_vectors)
    model.build_memory_from_data(corpus_vectors[:n])
    info = model.count_parameters()
    if seed == 42:
        print(f"  Params: {info['total_trainable']:,} | Memory: {info['memory_bank_size']:,} chunks ({info['memory_bank_bytes']/1024/1024:.1f}MB)")

    train_cfg = TrainConfig(
        seed=seed, batch_size=32, max_steps=5000,
        val_every=250, log_every=100, save_every=5000,
        warmup_steps=200, lr=3e-4,
    )
    tl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    best_val, _ = train(model, tl, vl, train_cfg,
                        run_name=f"corpus_mem_n{n}_s{seed}", tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()
    return best_val


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    t_cfg = TransformerConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    # Etape 1 : Charger le baseline pre-entraine
    print("Chargement du baseline pre-entraine...")
    baseline = BaselineTransformer(t_cfg)
    baseline.load_state_dict(torch.load("D:/Nature/runs/baseline_full_s42/best.pt",
                                        map_location='cpu', weights_only=True))
    device = torch.device('cuda')
    baseline = baseline.to(device)

    # Etape 2 : Encoder le corpus
    print("Encodage du corpus...")
    corpus_vectors = encode_corpus(baseline, train_ds, device, chunk_size=32)
    del baseline
    torch.cuda.empty_cache()

    # Sauvegarder les vecteurs pour reutilisation
    np.save("D:/Nature/runs/corpus_vectors.npy", corpus_vectors)
    print(f"Vecteurs sauvegardes: {corpus_vectors.shape}")

    # Etape 3 : Entrainer avec memoire de corpus
    results = []
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# CORPUS MEMORY seed {seed}")
        print(f"{'#'*60}")
        val = run_one(None, corpus_vectors, seed, train_ds, val_ds, tokenizer, t_cfg)
        results.append((seed, val))

    # Resume
    vals = np.array([v for _, v in results])
    bl6 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_full_s{s}/log.json'))['steps']]) for s in seeds])

    print(f"\n{'='*60}")
    print("RESULTATS MEMOIRE DE CORPUS")
    print(f"{'='*60}")
    print(f"  Baseline 6L:    {bl6.mean():.4f} +/- {bl6.std():.4f}")
    print(f"  Corpus memory:  {vals.mean():.4f} +/- {vals.std():.4f}  ({(vals.mean()-bl6.mean())/bl6.mean()*100:+.1f}%)")
    for s, v in results:
        print(f"    seed {s}: {v:.4f}")


if __name__ == "__main__":
    main()
