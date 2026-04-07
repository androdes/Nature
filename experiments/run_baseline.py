"""Experience 0 : Baseline transformer - verification de convergence."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.transformer import BaselineTransformer
from src.train import train


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    torch.manual_seed(seed)

    # Configs
    data_cfg = DataConfig()
    model_cfg = TransformerConfig()
    train_cfg = TrainConfig(
        seed=seed,
        batch_size=64,
        max_steps=500,  # Convergence check d'abord
        val_every=50,
        log_every=10,
    )

    # Data
    train_dataset, val_dataset, tokenizer = prepare_data(data_cfg)

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

    # Model
    model = BaselineTransformer(model_cfg)
    params = model.count_parameters()
    print(f"\nParametres: {params['total']:,} total, {params['trainable']:,} trainable")
    print(f"Config: {model_cfg.n_layers}L / {model_cfg.d_model}d / {model_cfg.n_heads}H")
    print(f"Batch size: {train_cfg.batch_size}, Seq len: {data_cfg.seq_len}")

    # VRAM estimation
    param_bytes = params['total'] * 4  # float32
    print(f"Params en memoire: {param_bytes / 1024**2:.1f} MB")

    # Train
    run_name = f"baseline_s{seed}_500steps"
    best_val, log = train(model, train_loader, val_loader, train_cfg,
                          run_name=run_name, tokenizer=tokenizer)

    print(f"\n{'='*60}")
    print(f"RESULTAT: seed={seed}, best_val_loss={best_val:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
