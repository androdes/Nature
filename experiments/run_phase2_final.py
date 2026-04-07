"""
Phase 2 final : deux tests de convergence.
1. 350M tout-VRAM (rapide, prouve la convergence d'un gros modele)
2. 1B offload (lent, prouve que l'extension fonctionne)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.offload_transformer import OffloadTransformer


def get_lr(step, config):
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    decay_ratio = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.lr * max(coeff, 0.1)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
        if n >= 30:
            break
    model.train()
    return total_loss / max(n, 1)


def train_model(model, train_ds, val_ds, config, run_name, use_offload_optim=False):
    device = torch.device('cuda')
    model.to_device(device)

    if use_offload_optim:
        model.setup_block_optimizers(lr=config.lr, weight_decay=config.weight_decay)
        optimizer = None
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay, betas=(0.9, 0.95),
        )

    run_dir = os.path.join(config.run_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    log = {'run_name': run_name, 'steps': []}
    train_iter = iter(train_loader)
    model.train()
    best_val = float('inf')
    t0 = time.time()

    p = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"{run_name}: {p['total']/1e6:.0f}M params, batch={config.batch_size}")
    print(f"offload_optim={use_offload_optim}")
    print(f"{'='*60}\n")

    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(step, config)
        if optimizer:
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        else:
            for opt in model._block_optimizers:
                for pg in opt.param_groups:
                    pg['lr'] = lr
            for pg in model._emb_optimizer.param_groups:
                pg['lr'] = lr

        _, loss = model(x, y)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if use_offload_optim:
            model.cpu_optimizer_step()
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % config.log_every == 0:
            dt = time.time() - t0
            ms = dt / config.log_every * 1000
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {ms:.0f}ms/step")
            t0 = time.time()

        if step % config.val_every == 0:
            val_loss = evaluate(model, val_loader, device)
            vram = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  >>> val_loss {val_loss:.4f} | vram {vram:.0f}MB")
            log['steps'].append({
                'step': step, 'train_loss': loss.item(),
                'val_loss': val_loss, 'lr': lr, 'peak_vram_mb': vram,
            })
            if val_loss < best_val:
                best_val = val_loss

    with open(os.path.join(run_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nBest val loss: {best_val:.4f}")
    return best_val


def main():
    data_cfg = DataConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Test 1 : 350M tout-VRAM (rapide)
    print("\n" + "#" * 60)
    print("# TEST 1 : 350M TOUT-VRAM")
    print("#" * 60)

    cfg_350 = TransformerConfig(n_layers=12, d_model=1536, n_heads=16, d_ff=6144, max_seq_len=256)
    model_350 = OffloadTransformer(cfg_350, use_checkpointing=True)
    train_cfg_350 = TrainConfig(
        seed=42, batch_size=8, max_steps=5000,
        val_every=500, log_every=50, save_every=5000,
        warmup_steps=200, lr=3e-4, grad_clip=1.0,
    )
    val_350 = train_model(model_350, train_ds, val_ds, train_cfg_350,
                          "phase2_350M_vram", use_offload_optim=False)
    del model_350
    torch.cuda.empty_cache()

    # Test 2 : 1B offload (lent, prouve l'extension)
    print("\n" + "#" * 60)
    print("# TEST 2 : 1B OFFLOAD")
    print("#" * 60)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    cfg_1b = TransformerConfig(n_layers=36, d_model=1536, n_heads=16, d_ff=6144, max_seq_len=256)
    model_1b = OffloadTransformer(cfg_1b, use_checkpointing=True)
    train_cfg_1b = TrainConfig(
        seed=42, batch_size=2, max_steps=2000,
        val_every=200, log_every=50, save_every=2000,
        warmup_steps=100, lr=3e-4, grad_clip=1.0,
    )
    val_1b = train_model(model_1b, train_ds, val_ds, train_cfg_1b,
                         "phase2_1B_offload", use_offload_optim=True)
    del model_1b
    torch.cuda.empty_cache()

    # Resume
    print(f"\n{'='*60}")
    print("PHASE 2 RESULTATS")
    print(f"{'='*60}")
    print(f"  350M tout-VRAM:  val_loss={val_350:.4f}")
    print(f"  1B offload:      val_loss={val_1b:.4f}")
    print(f"  Baseline 6L ref: ~4.711")


if __name__ == "__main__":
    main()
