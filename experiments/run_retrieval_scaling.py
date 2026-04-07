"""Exp 8 : Scaling de la memoire via FAISS retrieval.
Teste N = 512, 8000, 50000 elements.
La memoire est initialisee depuis les token embeddings (+ bruit pour N>8000).
K=64 elements recuperes a chaque forward pass."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.retrieval_memory import RetrievalMemoryTransformer
from src.train import train


def run_one(n_mem, seed, train_ds, val_ds, tokenizer, t_cfg):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = RetrievalMemoryTransformer(t_cfg, retrieve_k=64)
    model.build_memory_from_embeddings(n_elements=n_mem)
    info = model.count_parameters()
    if seed == 42:
        print(f"  Params: {info['total_trainable']:,} | Memory: {info['memory_bank_size']:,} ({info['memory_bank_bytes']/1024/1024:.1f}MB)")

    train_cfg = TrainConfig(
        seed=seed, batch_size=32, max_steps=5000,
        val_every=250, log_every=100, save_every=5000,
        warmup_steps=200, lr=3e-4,
    )
    tl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    best_val, _ = train(model, tl, vl, train_cfg,
                        run_name=f"retrieval_n{n_mem}_s{seed}", tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()
    return best_val


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    t_cfg = TransformerConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    mem_sizes = [512, 8000, 50000]
    all_results = {}

    for n_mem in mem_sizes:
        print(f"\n{'#'*60}")
        print(f"# RETRIEVAL N={n_mem}")
        print(f"{'#'*60}")
        results = []
        for seed in seeds:
            val = run_one(n_mem, seed, train_ds, val_ds, tokenizer, t_cfg)
            results.append((seed, val))
        all_results[n_mem] = results

    # Resume
    print(f"\n{'='*60}")
    print("SCALING MEMOIRE RETRIEVAL")
    print(f"{'='*60}")

    bl6 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_full_s{s}/log.json'))['steps']]) for s in seeds])
    print(f"\n  Baseline 6L: {bl6.mean():.4f} +/- {bl6.std():.4f}")

    for n_mem in mem_sizes:
        vals = np.array([v for _, v in all_results[n_mem]])
        delta = vals.mean() - bl6.mean()
        pct = delta / bl6.mean() * 100
        print(f"  Retrieval N={n_mem:>6d}: {vals.mean():.4f} +/- {vals.std():.4f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
