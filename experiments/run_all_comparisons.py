"""Lance sequentiellement: baseline 8L (3 seeds) puis AFMT (3 seeds)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, MemoryConfig, TrainConfig
from src.data import prepare_data
from src.transformer import BaselineTransformer
from src.afm_transformer import AFMTransformer
from src.train import train


def make_loaders(train_ds, val_ds, batch_size):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader


def run_experiment(model, train_ds, val_ds, tokenizer, seed, run_name, batch_size=64):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_cfg = TrainConfig(
        seed=seed, batch_size=batch_size, max_steps=5000,
        val_every=250, log_every=50, save_every=2500, warmup_steps=200, lr=3e-4,
    )
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size)
    best_val, log = train(model, train_loader, val_loader, train_cfg,
                          run_name=run_name, tokenizer=tokenizer)
    # Liberer la VRAM
    del model
    torch.cuda.empty_cache()
    return best_val


def print_results(name, results):
    vals = np.array([v for _, v in results])
    print(f"\n{name}: {vals.mean():.4f} +/- {vals.std():.4f}")
    for s, v in results:
        print(f"  seed {s}: {v:.4f}")


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    all_results = {}

    # --- Baseline 8L ---
    print(f"\n{'#'*60}")
    print("# EXPERIENCE: BASELINE 8L (controle params)")
    print(f"{'#'*60}")
    t_cfg_8L = TransformerConfig(n_layers=8)
    results_8L = []
    for seed in seeds:
        print(f"\n--- Baseline 8L seed {seed} ---")
        model = BaselineTransformer(t_cfg_8L)
        p = model.count_parameters()
        print(f"Params: {p['total']:,}")
        val = run_experiment(model, train_ds, val_ds, tokenizer, seed,
                            f"baseline_8L_s{seed}", batch_size=32)
        results_8L.append((seed, val))
    print_results("Baseline 8L", results_8L)
    all_results['baseline_8L'] = results_8L

    # --- AFMT ---
    print(f"\n{'#'*60}")
    print("# EXPERIENCE: AFMT (Activation Field Memory Transformer)")
    print(f"{'#'*60}")
    t_cfg = TransformerConfig()
    m_cfg = MemoryConfig(n_elements=2048, n_neighbors=32)
    results_afm = []
    for seed in seeds:
        print(f"\n--- AFMT seed {seed} ---")
        model = AFMTransformer(t_cfg, m_cfg)
        p = model.count_parameters()
        print(f"Params: {p['total']:,} (transformer: {p['transformer']:,}, memory: {p['memory_elements']:,})")
        val = run_experiment(model, train_ds, val_ds, tokenizer, seed,
                            f"afm_n2048_s{seed}", batch_size=32)
        results_afm.append((seed, val))
    print_results("AFMT", results_afm)
    all_results['AFMT'] = results_afm

    # --- Resume final ---
    print(f"\n{'='*60}")
    print("RESUME FINAL")
    print(f"{'='*60}")
    for name, results in all_results.items():
        vals = np.array([v for _, v in results])
        print(f"  {name:<20}: {vals.mean():.4f} +/- {vals.std():.4f}")


if __name__ == "__main__":
    main()
