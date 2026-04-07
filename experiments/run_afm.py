"""Experience 2 : AFMT (Activation Field Memory Transformer) - Phase 2 flat."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, MemoryConfig, TrainConfig
from src.data import prepare_data
from src.afm_transformer import AFMTransformer
from src.train import train


def run_seed(seed, train_dataset, val_dataset, tokenizer, t_cfg, m_cfg):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_cfg = TrainConfig(
        seed=seed,
        batch_size=64,
        max_steps=5000,
        val_every=250,
        log_every=50,
        save_every=2500,
        warmup_steps=200,
        lr=3e-4,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = AFMTransformer(t_cfg, m_cfg)
    params = model.count_parameters()
    print(f"\nParametres AFMT:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    print(f"Config transformer: {t_cfg.n_layers}L / {t_cfg.d_model}d / {t_cfg.n_heads}H")
    print(f"Config memoire: {m_cfg.n_elements} elements, {m_cfg.n_neighbors} voisins")

    run_name = f"afm_n{m_cfg.n_elements}_s{seed}"
    best_val, log = train(model, train_loader, val_loader, train_cfg,
                          run_name=run_name, tokenizer=tokenizer)
    return seed, best_val


def main():
    seeds = [42, 137, 256]
    if len(sys.argv) > 1:
        seeds = [int(s) for s in sys.argv[1:]]

    data_cfg = DataConfig()
    t_cfg = TransformerConfig()
    m_cfg = MemoryConfig(n_elements=4096, n_neighbors=32)

    train_dataset, val_dataset, tokenizer = prepare_data(data_cfg)

    results = []
    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# AFMT SEED {seed}")
        print(f"{'#'*60}")
        s, val = run_seed(seed, train_dataset, val_dataset, tokenizer, t_cfg, m_cfg)
        results.append((s, val))

    print(f"\n{'='*60}")
    print("RESULTATS AFMT")
    print(f"{'='*60}")
    vals = [v for _, v in results]
    for s, v in results:
        print(f"  seed {s}: val_loss = {v:.4f}")
    import numpy as np
    vals = np.array(vals)
    print(f"\n  Moyenne: {vals.mean():.4f} +/- {vals.std():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
