"""
Phase 2 : Rotation des couches transformer VRAM/RAM.

Test 1 : 12L/512d (42M) avec offloading — converge-t-il ?
Test 2 : 24L/512d (80M) — trop gros pour VRAM normale, possible avec offloading
Test 3 : si possible, 48L/512d (160M)

Baseline comparatif : 6L/256d (6.8M) en VRAM pure.
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


def train_offload(model, train_ds, val_ds, config, run_name, vram_reserve_mb=2048):
    device = torch.device('cuda')
    model.to_device(device)
    model.setup_block_optimizers(lr=config.lr, weight_decay=config.weight_decay)

    run_dir = os.path.join(config.run_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    log = {'config': config.__dict__, 'run_name': run_name, 'steps': []}
    train_iter = iter(train_loader)
    model.train()
    best_val = float('inf')
    t0 = time.time()
    model.reset_stats()

    p = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Total params: {p['total']:,} | Per block: {p['per_block']:,}")
    print(f"{'='*60}\n")

    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule — appliquer a tous les block optimizers
        lr = get_lr(step, config)
        for opt in model._block_optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr
        for pg in model._emb_optimizer.param_groups:
            pg['lr'] = lr

        _, loss = model(x, y)
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        model.cpu_optimizer_step()

        if step % config.log_every == 0:
            dt = time.time() - t0
            ms_per_step = dt / config.log_every * 1000
            stats = model.get_stats()
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {ms_per_step:.0f}ms/step | optim {stats['avg_optim_ms']:.0f}ms")
            t0 = time.time()
            model.reset_stats()

        if step % config.val_every == 0:
            val_loss = evaluate(model, val_loader, device)
            vram = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"  >>> val_loss {val_loss:.4f} | peak_vram {vram:.0f}MB")
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

    configs = [
        # (n_layers, d_model, n_heads, d_ff, batch, name)
        (24, 768, 12, 3072, 8, "offload_24L_768d_v3"),     # ~130M — tout VRAM
        (36, 1536, 16, 6144, 2, "offload_36L_1536d_v3"),   # ~1B — optimizer offload
    ]

    results = {}
    for n_layers, d_model, n_heads, d_ff, batch, name in configs:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        t_cfg = TransformerConfig(
            n_layers=n_layers, d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, max_seq_len=256,
        )
        train_cfg = TrainConfig(
            seed=42, batch_size=batch, max_steps=2000,
            val_every=200, log_every=20, save_every=2000,
            warmup_steps=100, lr=3e-4, grad_clip=1.0,
        )

        print(f"\n{'#'*60}")
        print(f"# {name} (batch={batch})")
        print(f"{'#'*60}")

        try:
            model = OffloadTransformer(t_cfg, use_checkpointing=True)
            p = model.count_parameters()
            print(f"Params: {p['total']:,} ({p['weight_mb']:.0f}MB)")

            val = train_offload(model, train_ds, val_ds, train_cfg, name)
            results[name] = {'val_loss': val, 'params': p['total'], 'ok': True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {'val_loss': None, 'params': 0, 'ok': False, 'error': str(e)}

        del model
        torch.cuda.empty_cache()

    # Resume
    print(f"\n{'='*60}")
    print("PHASE 2 : ROTATION DES COUCHES")
    print(f"{'='*60}")
    for name, r in results.items():
        if r['ok']:
            print(f"  {name}: {r['params']:,} params | val_loss={r['val_loss']:.4f}")
        else:
            print(f"  {name}: ECHEC — {r.get('error','?')[:60]}")


if __name__ == "__main__":
    main()
