"""Exp 10 : SimpleMem sur backbone 8L. L'effet est-il additif?
Baseline 8L: 4.648. SimpleMem 6L: 4.670. Si additif: ~4.59?"""

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


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    # 8L backbone + 512 elements, 1 cross-attn layer
    t_cfg = TransformerConfig(n_layers=8)
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n{'#'*60}")
        print(f"# 8L + SimpleMem seed {seed}")
        print(f"{'#'*60}")

        model = SimpleMemoryTransformer(t_cfg, n_mem_elements=512, n_mem_layers=1)
        p = model.count_parameters()
        print(f"Params: {p['total']:,}")

        train_cfg = TrainConfig(
            seed=seed, batch_size=32, max_steps=5000,
            val_every=250, log_every=100, save_every=5000,
            warmup_steps=200, lr=3e-4,
        )
        tl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
        vl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

        best_val, _ = train(model, tl, vl, train_cfg,
                            run_name=f"8L_smem_s{seed}", tokenizer=tokenizer)
        results.append((seed, best_val))
        del model
        torch.cuda.empty_cache()

    vals = np.array([v for _, v in results])
    bl6 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_full_s{s}/log.json'))['steps']]) for s in seeds])
    bl8 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_8L_s{s}/log.json'))['steps']]) for s in seeds])

    print(f"\n{'='*60}")
    print("RESULTATS 8L + SIMPLEMEM")
    print(f"{'='*60}")
    print(f"  Baseline 6L:     {bl6.mean():.4f} +/- {bl6.std():.4f}")
    print(f"  Baseline 8L:     {bl8.mean():.4f} +/- {bl8.std():.4f}")
    print(f"  8L + SimpleMem:  {vals.mean():.4f} +/- {vals.std():.4f}  ({(vals.mean()-bl6.mean())/bl6.mean()*100:+.1f}% vs 6L)")
    print(f"  Additif attendu: ~{bl8.mean() - (bl6.mean() - 4.670):.4f}")
    for s, v in results:
        print(f"    seed {s}: {v:.4f}")


if __name__ == "__main__":
    main()
