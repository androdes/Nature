"""Boucle d'entrainement pour Nature."""

import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.transformer import BaselineTransformer


def get_lr(step, config: TrainConfig):
    """Learning rate avec warmup lineaire puis cosine decay."""
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    decay_ratio = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.lr * max(coeff, 0.1)  # min 10% du lr


def _extract_loss(output):
    """Extrait la loss d'un forward pass (baseline retourne 2 vals, AFMT 3)."""
    if isinstance(output, tuple) and len(output) >= 2:
        return output[1]
    return output


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evalue la loss sur le dataset de validation."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        output = model(x, y)
        loss = _extract_loss(output)
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= 50:  # Cap a 50 batches pour la vitesse
            break
    model.train()
    return total_loss / max(n_batches, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    run_name: str = None,
    tokenizer=None,
):
    """Boucle d'entrainement principale."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"baseline_{timestamp}"
    run_dir = os.path.join(config.run_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Logging
    log = {
        'config': {
            'train': config.__dict__,
            'run_name': run_name,
        },
        'steps': [],
    }

    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}\n")

    train_iter = iter(train_loader)
    model.train()
    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(1, config.max_steps + 1):
        # Get batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward + backward
        output = model(x, y)
        loss = _extract_loss(output)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        # Recompute du graphe de propagation si applicable
        if hasattr(model, 'activation_field') and hasattr(model, 'm_config'):
            if step % model.m_config.recompute_graph_every == 0:
                model.activation_field.recompute_graph()

        # Logging
        if step % config.log_every == 0:
            dt = time.time() - t0
            tokens_per_sec = config.batch_size * train_loader.dataset.seq_len * config.log_every / dt
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {tokens_per_sec:.0f} tok/s")
            t0 = time.time()

        # Validation
        if step % config.val_every == 0:
            val_loss = evaluate(model, val_loader, device)
            step_log = {
                'step': step,
                'train_loss': loss.item(),
                'val_loss': val_loss,
                'lr': lr,
            }
            # Diagnostics du champ d'activation si disponible
            if isinstance(output, tuple) and len(output) >= 3 and output[2] is not None:
                act_diag = model.activation_field.get_diagnostics(output[2])
                step_log['activation'] = act_diag
                print(f"  >>> val_loss {val_loss:.4f} | act_mean {act_diag['act_mean']:.4f} sparsity {act_diag['act_sparsity']:.2f}")
            else:
                print(f"  >>> val_loss {val_loss:.4f}")

            log['steps'].append(step_log)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(run_dir, 'best.pt'))

            # Generation echantillon
            if tokenizer is not None:
                model.eval()
                prompt = torch.tensor([[tokenizer.encode("La").ids[0]]], device=device)
                generated = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_k=40)
                text = tokenizer.decode(generated[0].tolist())
                print(f"  >>> sample: {text[:200].encode('ascii', errors='replace').decode()}")
                model.train()

        # Save checkpoint
        if step % config.save_every == 0:
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss if step % config.val_every == 0 else None,
            }, os.path.join(run_dir, f'ckpt_{step}.pt'))

    # Sauvegarde finale
    torch.save(model.state_dict(), os.path.join(run_dir, 'final.pt'))
    with open(os.path.join(run_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Saved to: {run_dir}")
    return best_val_loss, log
