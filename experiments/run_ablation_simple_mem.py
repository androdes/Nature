"""Exp 6 : Ablation - memoire externe simple (sans activation field).
Teste si la memoire aide du tout, independamment du champ d'activation.
Deux regimes : grand modele (6L/256d) et petit modele (2L/128d)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.simple_memory_transformer import SimpleMemoryTransformer
from src.train import train


def run_one(model, train_ds, val_ds, tokenizer, seed, run_name, batch_size=64, lr=3e-4):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_cfg = TrainConfig(
        seed=seed, batch_size=batch_size, max_steps=5000,
        val_every=250, log_every=50, save_every=5000,
        warmup_steps=200, lr=lr,
    )
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    best_val, log = train(model, tl, vl, train_cfg, run_name=run_name, tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()
    return best_val


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    # --- Regime petit : 2L/128d + 256 elements memoire, 1 cross-attn ---
    print(f"\n{'#'*60}")
    print("# SIMPLE MEMORY TINY (2L/128d + 256 mem, 1 layer)")
    print(f"{'#'*60}")
    tiny_cfg = TransformerConfig(n_layers=2, d_model=128, n_heads=4, d_ff=512, max_seq_len=256)
    results_tiny = []
    for seed in seeds:
        model = SimpleMemoryTransformer(tiny_cfg, n_mem_elements=256, n_mem_layers=1)
        p = model.count_parameters()
        print(f"\nseed {seed} | Params: {p['total']:,} (mem: {p['memory']:,})")
        val = run_one(model, train_ds, val_ds, tokenizer, seed,
                      f"simple_mem_tiny_s{seed}", batch_size=128, lr=5e-4)
        results_tiny.append((seed, val))

    # --- Regime grand : 6L/256d + 512 elements memoire, 1 cross-attn ---
    print(f"\n{'#'*60}")
    print("# SIMPLE MEMORY LARGE (6L/256d + 512 mem, 1 layer)")
    print(f"{'#'*60}")
    large_cfg = TransformerConfig()
    results_large = []
    for seed in seeds:
        model = SimpleMemoryTransformer(large_cfg, n_mem_elements=512, n_mem_layers=1)
        p = model.count_parameters()
        print(f"\nseed {seed} | Params: {p['total']:,} (mem: {p['memory']:,})")
        val = run_one(model, train_ds, val_ds, tokenizer, seed,
                      f"simple_mem_large_s{seed}", batch_size=32, lr=3e-4)
        results_large.append((seed, val))

    # --- Resume ---
    print(f"\n{'='*60}")
    print("RESULTATS ABLATION SIMPLE MEMORY")
    print(f"{'='*60}")

    for name, results in [("SimpleMem tiny", results_tiny), ("SimpleMem large", results_large)]:
        vals = np.array([v for _, v in results])
        print(f"\n  {name}: {vals.mean():.4f} +/- {vals.std():.4f}")
        for s, v in results:
            print(f"    seed {s}: {v:.4f}")

    # Comparaisons
    vt = np.array([v for _, v in results_tiny])
    vl_arr = np.array([v for _, v in results_large])

    # References
    try:
        bl2 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/tiny_baseline_s{s}/log.json'))['steps']]) for s in seeds])
        print(f"\n  Ref Baseline 2L: {bl2.mean():.4f} +/- {bl2.std():.4f}")
        print(f"  SimpleMem tiny vs 2L: {vt.mean()-bl2.mean():+.4f} ({(vt.mean()-bl2.mean())/bl2.mean()*100:+.1f}%)")
    except: pass

    try:
        bl6 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_full_s{s}/log.json'))['steps']]) for s in seeds])
        print(f"\n  Ref Baseline 6L: {bl6.mean():.4f} +/- {bl6.std():.4f}")
        print(f"  SimpleMem large vs 6L: {vl_arr.mean()-bl6.mean():+.4f} ({(vl_arr.mean()-bl6.mean())/bl6.mean()*100:+.1f}%)")
    except: pass


if __name__ == "__main__":
    main()
