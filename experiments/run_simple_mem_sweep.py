"""Exp 7 : Sweep sur la memoire simple - trouver la config optimale.
Varie : N (128, 256, 512, 1024), n_layers (1, 2), placement."""

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


def run_one(model, train_ds, val_ds, tokenizer, seed, run_name, batch_size=32):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_cfg = TrainConfig(
        seed=seed, batch_size=batch_size, max_steps=5000,
        val_every=250, log_every=100, save_every=5000,
        warmup_steps=200, lr=3e-4,
    )
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    best_val, _ = train(model, tl, vl, train_cfg, run_name=run_name, tokenizer=tokenizer)
    del model
    torch.cuda.empty_cache()
    return best_val


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    large_cfg = TransformerConfig()  # 6L/256d/4H
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    configs = [
        # (n_elements, n_mem_layers, name)
        (128, 1, "n128_L1"),
        (256, 1, "n256_L1"),
        (512, 1, "n512_L1"),   # le gagnant de exp 6
        (1024, 1, "n1024_L1"),
        (512, 2, "n512_L2"),
        (512, 3, "n512_L3"),
    ]

    all_results = {}

    for n_elem, n_layers, name in configs:
        print(f"\n{'#'*60}")
        print(f"# SimpleMem {name}")
        print(f"{'#'*60}")
        results = []
        for seed in seeds:
            model = SimpleMemoryTransformer(large_cfg, n_mem_elements=n_elem, n_mem_layers=n_layers)
            p = model.count_parameters()
            if seed == seeds[0]:
                print(f"Params: {p['total']:,} (mem: {p['memory']:,})")
            val = run_one(model, train_ds, val_ds, tokenizer, seed,
                          f"smem_{name}_s{seed}")
            results.append((seed, val))
        all_results[name] = results

    # Resume
    print(f"\n{'='*60}")
    print("SWEEP RESULTATS")
    print(f"{'='*60}")

    bl6 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_full_s{s}/log.json'))['steps']]) for s in seeds])
    print(f"\n  Baseline 6L: {bl6.mean():.4f} +/- {bl6.std():.4f}")
    print()

    for name, results in sorted(all_results.items(), key=lambda x: np.mean([v for _, v in x[1]])):
        vals = np.array([v for _, v in results])
        delta = vals.mean() - bl6.mean()
        pct = delta / bl6.mean() * 100
        print(f"  {name:15s}: {vals.mean():.4f} +/- {vals.std():.4f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
