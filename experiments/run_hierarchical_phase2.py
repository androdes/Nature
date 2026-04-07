"""
Phase 2 : La tortue va plus loin.

Meme VRAM. Memoire totale croissante : 5K, 50K, 500K elements.
A chaque taille, on verifie que le systeme tient et apprend.
Metrique principale : la plus grande taille ou le systeme apprend.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import time
import numpy as np
from torch.utils.data import DataLoader

from src.config import DataConfig, TransformerConfig, TrainConfig
from src.data import prepare_data
from src.hierarchical_memory import HierarchicalMemoryTransformer

# Reutiliser la boucle de phase 1
from experiments.run_hierarchical_phase1 import train_hierarchical


def main():
    seed = 42  # Un seul seed par taille pour le scaling test
    data_cfg = DataConfig()
    t_cfg = TransformerConfig()
    train_ds, val_ds, tokenizer = prepare_data(data_cfg)

    # Tailles croissantes
    # Reference VRAM : 512 elements = 512*256*4 = 0.5 MB
    # 5K = 10x, 50K = 100x, 500K = 1000x
    sizes = [5000, 50000, 500000]

    train_cfg = TrainConfig(
        seed=seed, batch_size=32, max_steps=5000, val_every=250,
        log_every=50, save_every=5000, warmup_steps=200, lr=3e-4,
    )

    results = {}

    for n_mem in sizes:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        mem_mb = n_mem * t_cfg.d_model * 4 / 1024 / 1024
        ratio = n_mem / 512

        print(f"\n{'#'*60}")
        print(f"# N={n_mem:,} ({ratio:.0f}x VRAM ref) — {mem_mb:.1f} MB sur disque")
        print(f"{'#'*60}")

        storage = f"D:/Nature/runs/membank_p2_n{n_mem}.dat"
        # RAM cache = 10% de N ou 50K max
        ram_cache = min(n_mem // 10, 50000)

        model = HierarchicalMemoryTransformer(
            t_cfg, n_mem_elements=n_mem, retrieve_k=64,
            ram_cache_size=ram_cache, storage_path=storage,
        )

        t0 = time.time()
        model.initialize_memory()
        init_time = time.time() - t0
        print(f"Init: {init_time:.1f}s")

        info = model.count_parameters()
        print(f"VRAM params: {info['trainable_vram']:,}")
        print(f"Memory: {info['memory_elements']:,} ({info['memory_bytes']/1024/1024:.1f}MB)")
        print(f"RAM cache: {ram_cache:,}")

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                num_workers=0, pin_memory=True)

        # Rebuild moins souvent pour les gros index
        rebuild_every = max(200, n_mem // 250)

        try:
            val = train_hierarchical(
                model, train_loader, val_loader, train_cfg,
                run_name=f"hier_n{n_mem}_s{seed}",
                mem_lr=1e-3, rebuild_every=rebuild_every,
            )
            stats = model.memory_bank.get_stats()
            results[n_mem] = {
                'val_loss': val,
                'converged': True,
                'ram_hit_rate': stats['ram_hit_rate'],
                'coverage': stats['coverage'],
                'unique_accessed': stats['unique_accessed'],
                'avg_retrieval_ms': stats['avg_retrieval_ms'],
            }
        except Exception as e:
            print(f"ECHEC a N={n_mem}: {e}")
            results[n_mem] = {'val_loss': None, 'converged': False, 'error': str(e)}

        del model
        torch.cuda.empty_cache()

        # Nettoyer le fichier memmap (gros)
        try:
            os.remove(storage)
        except:
            pass

    # Resume
    print(f"\n{'='*60}")
    print("PHASE 2 : SCALING HIERARCHIQUE")
    print(f"{'='*60}")

    # Reference phase 1
    try:
        d = json.load(open('D:/Nature/runs/hier_n512_s42/log.json'))
        ref_val = min([x['val_loss'] for x in d['steps']])
        print(f"\n  Reference (N=512): {ref_val:.4f}")
    except:
        ref_val = None

    print(f"\n  {'N':>8s} | {'Ratio':>6s} | {'Val Loss':>9s} | {'RAM Hit':>7s} | {'Coverage':>8s} | {'Retr ms':>7s} | {'Status'}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}")

    for n_mem in sizes:
        r = results.get(n_mem, {})
        if r.get('converged'):
            print(f"  {n_mem:>8,} | {n_mem/512:>5.0f}x | {r['val_loss']:>9.4f} | {r['ram_hit_rate']:>6.1%} | {r['coverage']:>7.3f} | {r['avg_retrieval_ms']:>6.1f} | OK")
        else:
            print(f"  {n_mem:>8,} | {n_mem/512:>5.0f}x | {'N/A':>9s} | {'N/A':>7s} | {'N/A':>8s} | {'N/A':>7s} | ECHEC: {r.get('error','?')[:40]}")

    # Metrique principale
    max_working = max([n for n, r in results.items() if r.get('converged')], default=0)
    print(f"\n  >>> Plus grande memoire fonctionnelle : {max_working:,} elements ({max_working*256*4/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
