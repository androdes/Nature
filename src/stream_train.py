"""
Boucle d'entrainement en flux (document par document) pour le transformer
a memoire vivante. La cave persiste d'un pas a l'autre ; elle n'est videe
qu'au changement de document de chaque flux.
"""

import os
import json
import math
import time
import torch
import torch.nn.functional as F


def get_lr(step, lr, warmup, max_steps):
    if step < warmup:
        return lr * step / warmup
    ratio = (step - warmup) / max(1, max_steps - warmup)
    return lr * max(0.5 * (1 + math.cos(math.pi * ratio)), 0.1)


@torch.no_grad()
def evaluate(model, batches, device, cat_names, mem_size, vram_slots, caviste_kwargs=None):
    """Loss et precision par categorie de token, en lisant chaque document dans l'ordre."""
    model.eval()
    n_streams = batches[0][0].shape[0]
    state = model.new_state(n_streams, mem_size, vram_slots, caviste_kwargs)
    tot_loss = {c: 0.0 for c in cat_names}
    tot_ok = {c: 0 for c in cat_names}
    tot_n = {c: 0 for c in cat_names}
    for x, y, cat, reset in batches:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, state=state, reset=reset.to(device))
        loss = F.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction='none').cpu()
        ok = (logits.argmax(-1) == y).cpu()
        mask = (y != -100).cpu()
        for c in cat_names:
            m = mask & (cat == c)
            tot_loss[c] += float(loss[m].sum())
            tot_ok[c] += int(ok[m].sum())
            tot_n[c] += int(m.sum())
    model.train()

    out = {}
    all_n = sum(tot_n.values())
    out['loss'] = sum(tot_loss.values()) / max(all_n, 1)
    for c, name in cat_names.items():
        n = tot_n[c]
        out[name] = {
            'n': n,
            'loss': tot_loss[c] / n if n else None,
            'acc': tot_ok[c] / n if n else None,
        }
    if state is not None:
        out['caviste'] = state.caviste.summary()
    return out


def fmt_eval(ev, cat_names):
    parts = [f"loss {ev['loss']:.4f}"]
    for name in cat_names.values():
        e = ev[name]
        if e['n']:
            parts.append(f"{name} {e['loss']:.3f}/{100 * e['acc']:.1f}%")
    if 'caviste' in ev:
        cv = ev['caviste']
        parts.append(f"VRAM caviste {100 * cv['caviste_hit_rate']:.1f}% vs LRU {100 * cv['lru_hit_rate']:.1f}%")
    return ' | '.join(parts)


def train_streams(model, streams, eval_batches, cat_names, *, steps, lr=3e-4,
                  warmup=200, weight_decay=0.01, grad_clip=1.0, val_every=250,
                  mem_size=16384, vram_slots=2048, caviste_kwargs=None,
                  device='cpu', run_dir=None, run_name='run', extra_config=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    state = model.new_state(streams.B, mem_size, vram_slots, caviste_kwargs)

    log = {
        'config': {
            'run_name': run_name, 'use_memory': model.use_memory, 'steps': steps,
            'lr': lr, 'mem_size': mem_size, 'vram_slots': vram_slots,
            'topk': model.topk, 'mem_layer': model.mem_layer,
            'params': model.count_parameters()['total'], **(extra_config or {}),
        },
        'steps': [],
    }
    print(f"\n{'=' * 60}\n{run_name} | memoire={'oui' if model.use_memory else 'non'} "
          f"| {log['config']['params']:,} params | device {device}\n{'=' * 60}")

    best = None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y, reset = streams.next_batch()
        x, y, reset = x.to(device), y.to(device), reset.to(device)
        cur_lr = get_lr(step, lr, warmup, steps)
        for g in opt.param_groups:
            g['lr'] = cur_lr

        _, loss = model(x, y, state=state, reset=reset)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step % val_every == 0 or step == steps:
            ev = evaluate(model, eval_batches, device, cat_names, mem_size, vram_slots, caviste_kwargs)
            entry = {'step': step, 'train_loss': loss.item(), 'lr': cur_lr,
                     'sec': round(time.time() - t0, 1), 'val': ev}
            if model.use_memory:
                entry['gate'] = [round(float(g), 3) for g in model.gate()]
            log['steps'].append(entry)
            if best is None or ev['loss'] < best['val']['loss']:
                best = entry
            gate = f" | porte {entry['gate']}" if 'gate' in entry else ''
            print(f"step {step:5d} | train {loss.item():.4f} | {fmt_eval(ev, cat_names)}{gate}")

    log['best'] = best
    log['final'] = log['steps'][-1]
    if run_dir:
        path = os.path.join(run_dir, run_name)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'log.json'), 'w') as f:
            json.dump(log, f, indent=2)
    return log
