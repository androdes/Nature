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


# Validation : meme jeu pour tous les modeles (tout le split val, batch fixe),
# independamment du batch d'entrainement. Sinon les val loss ne sont pas comparables.
EVAL_BATCH_SIZE = 8


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Loss moyenne par sequence sur l'integralite du split de validation."""
    model.eval()
    total_loss, n_seq = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item() * x.size(0)
        n_seq += x.size(0)
    model.train()
    return total_loss / max(n_seq, 1)


def save_log(log, run_dir):
    with open(os.path.join(run_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)


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
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    accum = max(1, config.accum_steps)
    seq_len = train_ds.seq_len
    log = {
        'run_name': run_name,
        'batch_size': config.batch_size,
        'accum_steps': accum,
        'effective_batch_size': config.batch_size * accum,
        'val_sequences': len(val_ds),
        'steps': [],
    }
    train_iter = iter(train_loader)
    model.train()
    best_val = float('inf')
    best_step = 0
    t0 = time.time()

    p = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"{run_name}: {p['total']/1e6:.0f}M params, batch={config.batch_size} "
          f"x accum={accum} (effectif {config.batch_size * accum})")
    print(f"offload_optim={use_offload_optim} | val sur {len(val_ds)} sequences")
    print(f"{'='*60}\n")

    for step in range(1, config.max_steps + 1):
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

        # Accumulation : plusieurs forward/backward pour un seul step optimiseur.
        # Les gradients sont remis a zero apres chaque step (les deux chemins).
        train_loss = 0.0
        for _ in range(accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            (loss / accum).backward()
            train_loss += loss.item() / accum

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
            print(f"step {step:5d} | loss {train_loss:.4f} | lr {lr:.2e} | "
                  f"{ms:.0f}ms/step | {ms / accum:.0f}ms/micro-batch")
            t0 = time.time()

        if step % config.val_every == 0 or step == config.max_steps:
            t_val = time.time()
            val_loss = evaluate(model, val_loader, device)
            vram = torch.cuda.max_memory_allocated() / 1024 / 1024
            tokens_seen = step * accum * config.batch_size * seq_len
            print(f"  >>> val_loss {val_loss:.4f} | vram {vram:.0f}MB | "
                  f"{tokens_seen / 1e6:.2f}M tokens vus")
            log['steps'].append({
                'step': step, 'train_loss': train_loss,
                'val_loss': val_loss, 'lr': lr, 'peak_vram_mb': vram,
                'tokens_seen': tokens_seen,
            })
            if val_loss < best_val:
                best_val = val_loss
                best_step = step
                # Poids seuls (state_dict brut), meme format que les autres runs.
                torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))
                print(f"  >>> best.pt sauvegarde (step {step})")
            log['best_val_loss'] = best_val
            log['best_step'] = best_step
            # Ecrit a chaque validation : un crash ne perd pas l'historique.
            save_log(log, run_dir)
            # Ne pas compter la validation dans les ms/step.
            t0 += time.time() - t_val

    save_log(log, run_dir)

    print(f"\nBest val loss: {best_val:.4f} (step {best_step})")
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
    # Le step optimiseur offloade (~8 s) domine le temps : on l'amortit sur 16
    # micro-batchs. 1000 steps x 16 x 2 x 256 = 8.2M tokens (~3 passes sur le corpus),
    # contre 1M tokens avec l'ancien reglage (2000 steps, batch 2, sans accumulation).
    train_cfg_1b = TrainConfig(
        seed=42, batch_size=2, accum_steps=16, max_steps=1000,
        val_every=100, log_every=25, save_every=1000,
        warmup_steps=50, lr=3e-4, grad_clip=1.0,
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
