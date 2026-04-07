"""Exp 5 : Petit modele (2L/128d) - regime sous-dimensionne.
0.2 params/token au lieu de 2.6. Le modele ne peut pas memoriser.
La memoire externe devrait apporter de la valeur ici."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, MemoryConfig, TrainConfig
from src.data import prepare_data
from src.transformer import BaselineTransformer
from src.afm_transformer import AFMTransformer
from src.train import train


def run_one(model, train_ds, val_ds, tokenizer, seed, run_name, batch_size=64):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_cfg = TrainConfig(
        seed=seed, batch_size=batch_size, max_steps=5000,
        val_every=250, log_every=50, save_every=5000,
        warmup_steps=200, lr=5e-4,  # LR plus haut pour petit modele
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    best_val, log = train(model, train_loader, val_loader, train_cfg,
                          run_name=run_name, tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()
    return best_val


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    # Tiny transformer : 2L/128d/4H -> ~530K params
    tiny_cfg = TransformerConfig(n_layers=2, d_model=128, n_heads=4, d_ff=512, max_seq_len=256)

    # --- Baseline tiny ---
    print(f"\n{'#'*60}")
    print("# BASELINE TINY (2L/128d)")
    print(f"{'#'*60}")
    results_bl = []
    for seed in seeds:
        model = BaselineTransformer(tiny_cfg)
        p = model.count_parameters()
        print(f"\nseed {seed} | Params: {p['total']:,}")
        val = run_one(model, train_ds, val_ds, tokenizer, seed,
                      f"tiny_baseline_s{seed}", batch_size=128)
        results_bl.append((seed, val))

    # --- AFMT tiny ---
    print(f"\n{'#'*60}")
    print("# AFMT TINY (2L/128d + 2048 mem K=64)")
    print(f"{'#'*60}")
    # Memoire avec d_model=128
    m_cfg = MemoryConfig(n_elements=2048, focus_k=64, n_neighbors=32)
    results_afm = []
    for seed in seeds:
        model = AFMTransformer(tiny_cfg, m_cfg)
        p = model.count_parameters()
        print(f"\nseed {seed} | Params: {p['total']:,}")
        val = run_one(model, train_ds, val_ds, tokenizer, seed,
                      f"tiny_afm_k64_s{seed}", batch_size=128)
        results_afm.append((seed, val))

    # --- Baseline avec params comparables : 4L/128d ---
    print(f"\n{'#'*60}")
    print("# BASELINE MATCHED (4L/128d - params comparables)")
    print(f"{'#'*60}")
    matched_cfg = TransformerConfig(n_layers=4, d_model=128, n_heads=4, d_ff=512, max_seq_len=256)
    results_match = []
    for seed in seeds:
        model = BaselineTransformer(matched_cfg)
        p = model.count_parameters()
        print(f"\nseed {seed} | Params: {p['total']:,}")
        val = run_one(model, train_ds, val_ds, tokenizer, seed,
                      f"tiny_matched_s{seed}", batch_size=128)
        results_match.append((seed, val))

    # --- Resume ---
    print(f"\n{'='*60}")
    print("RESULTATS EXP 5 : REGIME SOUS-DIMENSIONNE")
    print(f"{'='*60}")
    for name, results in [("Baseline 2L", results_bl), ("AFMT 2L+mem", results_afm), ("Baseline 4L", results_match)]:
        vals = np.array([v for _, v in results])
        print(f"\n  {name}: {vals.mean():.4f} +/- {vals.std():.4f}")
        for s, v in results:
            print(f"    seed {s}: {v:.4f}")

    vbl = np.array([v for _, v in results_bl])
    vafm = np.array([v for _, v in results_afm])
    vmatch = np.array([v for _, v in results_match])
    print(f"\n  AFMT vs baseline 2L: {vafm.mean()-vbl.mean():+.4f} ({(vafm.mean()-vbl.mean())/vbl.mean()*100:+.1f}%)")
    print(f"  AFMT vs baseline 4L: {vafm.mean()-vmatch.mean():+.4f} ({(vafm.mean()-vmatch.mean())/vmatch.mean()*100:+.1f}%)")


if __name__ == "__main__":
    main()
