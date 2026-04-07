"""AFMT v2 : FocusCrossAttention avec selection content-based du foyer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, MemoryConfig, TrainConfig
from src.data import prepare_data
from src.afm_transformer import AFMTransformer
from src.train import train


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    t_cfg = TransformerConfig()
    m_cfg = MemoryConfig(n_elements=2048, focus_k=64, n_neighbors=32)

    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n{'#'*60}")
        print(f"# AFMT v2 SEED {seed}")
        print(f"{'#'*60}")

        model = AFMTransformer(t_cfg, m_cfg)
        p = model.count_parameters()
        print(f"Params: {p['total']:,}")

        train_cfg = TrainConfig(
            seed=seed, batch_size=32, max_steps=5000,
            val_every=250, log_every=50, save_every=2500,
            warmup_steps=200, lr=3e-4,
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                num_workers=0, pin_memory=True)

        best_val, log = train(model, train_loader, val_loader, train_cfg,
                              run_name=f"afm_v2_k64_s{seed}", tokenizer=tokenizer)
        results.append((seed, best_val))

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("RESULTATS AFMT v2")
    print(f"{'='*60}")
    vals = np.array([v for _, v in results])
    for s, v in results:
        print(f"  seed {s}: val_loss = {v:.4f}")
    print(f"\n  Moyenne: {vals.mean():.4f} +/- {vals.std():.4f}")

    # Comparaison avec baselines
    import json
    bl6 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_full_s{s}/log.json'))['steps']]) for s in [42,137,256]])
    bl8 = np.array([min([x['val_loss'] for x in json.load(open(f'D:/Nature/runs/baseline_8L_s{s}/log.json'))['steps']]) for s in [42,137,256]])
    print(f"\n  Baseline 6L: {bl6.mean():.4f} +/- {bl6.std():.4f}")
    print(f"  Baseline 8L: {bl8.mean():.4f} +/- {bl8.std():.4f}")
    print(f"  AFMT v2:     {vals.mean():.4f} +/- {vals.std():.4f}")
    print(f"  Delta vs 6L: {vals.mean()-bl6.mean():+.4f} ({(vals.mean()-bl6.mean())/bl6.mean()*100:+.1f}%)")
    print(f"  Delta vs 8L: {vals.mean()-bl8.mean():+.4f} ({(vals.mean()-bl8.mean())/bl8.mean()*100:+.1f}%)")


if __name__ == "__main__":
    main()
