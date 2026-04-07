"""
Phase 1 : La mecanique tient.

Compare :
  Version A : SimpleMem (tout en VRAM, N=512)
  Version B : HierarchicalMem (N=512, meme taille, mais reparti VRAM/RAM/disque)

Critere : B converge avec val_loss dans +/-5% de A.
Le temps peut etre 10x plus long.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.simple_memory_transformer import SimpleMemoryTransformer
from src.hierarchical_memory import HierarchicalMemoryTransformer


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
        if n >= 50:
            break
    model.train()
    return total_loss / max(n, 1)


def train_hierarchical(model, train_loader, val_loader, config, run_name,
                       mem_lr=1e-3, rebuild_every=200):
    """Boucle d'entrainement avec gestion des gradients memoire."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_dir = os.path.join(config.run_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Optimizer pour les params transformer (pas la memoire externe)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr,
        weight_decay=config.weight_decay, betas=(0.9, 0.95),
    )

    log = {'config': config.__dict__, 'run_name': run_name, 'steps': []}
    train_iter = iter(train_loader)
    model.train()
    best_val = float('inf')
    t0 = time.time()
    total_mem_updates = 0

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Device: {device}")
    info = model.count_parameters()
    print(f"VRAM params: {info['trainable_vram']:,} | Memory: {info['memory_elements']:,} ({info['memory_bytes']/1024/1024:.1f}MB)")
    print(f"{'='*60}\n")

    for step in range(1, config.max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        _, loss = model(x, y)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Appliquer les gradients a la memoire externe
        if hasattr(model, 'apply_memory_gradients'):
            n_upd = model.apply_memory_gradients(lr=mem_lr)
            total_mem_updates += n_upd

        # Rebuild index periodiquement
        if hasattr(model, 'rebuild_index') and step % rebuild_every == 0:
            model.rebuild_index()

        # Logging
        if step % config.log_every == 0:
            dt = time.time() - t0
            tok_s = config.batch_size * train_loader.dataset.seq_len * config.log_every / dt
            extra = ""
            if hasattr(model, 'memory_bank'):
                stats = model.memory_bank.get_stats()
                extra = f" | ram_hit {stats['ram_hit_rate']:.2f} retr {stats['avg_retrieval_ms']:.1f}ms cov {stats['coverage']:.3f}"
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tok_s:.0f} tok/s{extra}")
            t0 = time.time()

        # Validation
        if step % config.val_every == 0:
            val_loss = evaluate(model, val_loader, device)
            step_log = {
                'step': step, 'train_loss': loss.item(),
                'val_loss': val_loss, 'lr': lr,
            }
            if hasattr(model, 'memory_bank'):
                step_log['mem_stats'] = model.memory_bank.get_stats()
                step_log['mem_stats']['unique_accessed'] = len(model.memory_bank.stats['unique_elements_accessed'])
            log['steps'].append(step_log)
            print(f"  >>> val_loss {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))

    # Sauvegarde finale
    torch.save(model.state_dict(), os.path.join(run_dir, 'final.pt'))
    with open(os.path.join(run_dir, 'log.json'), 'w') as f:
        # Convertir les sets en listes pour JSON
        for s in log['steps']:
            if 'mem_stats' in s:
                s['mem_stats'].pop('unique_elements_accessed', None)
        json.dump(log, f, indent=2)

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Total memory updates: {total_mem_updates:,}")
    return best_val


def main():
    seeds = [42, 137, 256]
    data_cfg = DataConfig()
    t_cfg = TransformerConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    train_cfg = TrainConfig(
        batch_size=32, max_steps=5000, val_every=250,
        log_every=50, save_every=5000, warmup_steps=200, lr=3e-4,
    )

    # --- Version A : SimpleMem tout en VRAM (reference) ---
    # On reutilise les resultats existants si disponibles
    print("\n=== VERSION A : SimpleMem VRAM (reference existante) ===")
    ref_vals = []
    for s in seeds:
        try:
            d = json.load(open(f'D:/Nature/runs/smem_n512_L1_s{s}/log.json'))
            v = min([x['val_loss'] for x in d['steps']])
            ref_vals.append(v)
            print(f"  s{s}: {v:.4f} (existant)")
        except:
            print(f"  s{s}: pas de resultat existant")
    ref = np.array(ref_vals)
    print(f"  Version A mean: {ref.mean():.4f} +/- {ref.std():.4f}")

    # --- Version B : Hierarchique (meme N=512, reparti) ---
    print("\n=== VERSION B : Hierarchique N=512 ===")
    results_b = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print(f"\n{'#'*60}")
        print(f"# Hierarchique N=512 seed {seed}")
        print(f"{'#'*60}")

        storage = f"D:/Nature/runs/membank_p1_s{seed}.dat"
        model = HierarchicalMemoryTransformer(
            t_cfg, n_mem_elements=512, retrieve_k=64,
            ram_cache_size=256, storage_path=storage,
        )
        model.initialize_memory()

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                num_workers=0, pin_memory=True)

        val = train_hierarchical(
            model, train_loader, val_loader, train_cfg,
            run_name=f"hier_n512_s{seed}", mem_lr=1e-3, rebuild_every=200,
        )
        results_b.append((seed, val))
        del model
        torch.cuda.empty_cache()

    # --- Comparaison ---
    vb = np.array([v for _, v in results_b])
    print(f"\n{'='*60}")
    print("PHASE 1 : COMPARAISON A vs B")
    print(f"{'='*60}")
    print(f"  Version A (VRAM):        {ref.mean():.4f} +/- {ref.std():.4f}")
    print(f"  Version B (hierarchique): {vb.mean():.4f} +/- {vb.std():.4f}")
    delta_pct = abs(vb.mean() - ref.mean()) / ref.mean() * 100
    print(f"  Ecart: {delta_pct:.1f}%")
    if delta_pct < 5:
        print(f"  >>> PHASE 1 VALIDEE (ecart < 5%)")
    else:
        print(f"  >>> PHASE 1 ECHOUEE (ecart >= 5%)")


if __name__ == "__main__":
    main()
