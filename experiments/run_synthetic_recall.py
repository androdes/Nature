"""
Le banquet avec allergie : la cave est-elle indispensable ?

Tache synthetique (src/synthetic_recall.py) ou chaque document enonce des
faits cle -> valeur au debut, puis pose des questions plusieurs segments
plus loin (proches : meme segment ; lointaines : dans la cave). Compare, seed par seed, le meme modele avec et sans cave.

    python experiments/run_synthetic_recall.py                 # 3 seeds, ~quelques minutes par run sur CPU
    python experiments/run_synthetic_recall.py --seeds 42 --steps 1500
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import TransformerConfig
from src.memorizing_transformer import MemorizingTransformer
from src.synthetic_recall import RecallTask, SyntheticStreams, CAT_NAMES
from src.stream_data import build_eval_batches
from src.stream_train import train_streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 137, 256])
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--streams', type=int, default=16)
    ap.add_argument('--seq-len', type=int, default=64)
    ap.add_argument('--mem-size', type=int, default=1024)
    ap.add_argument('--vram-slots', type=int, default=128)
    ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-docs', type=int, default=64)
    ap.add_argument('--run-dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs'))
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task = RecallTask(seq_len=args.seq_len)
    eval_rng = np.random.default_rng(999)
    pairs = [task.sample(eval_rng) for _ in range(args.eval_docs)]
    eval_batches = build_eval_batches([d for d, _ in pairs], args.streams, args.seq_len,
                                      cats=[c for _, c in pairs])
    cfg = TransformerConfig(vocab_size=task.vocab_size, n_layers=4, n_heads=4, d_model=128,
                            d_ff=512, max_seq_len=args.seq_len, dropout=0.0)

    results = {}
    for seed in args.seeds:
        for use_memory in (False, True):
            torch.manual_seed(seed)
            model = MemorizingTransformer(cfg, mem_layer=2, topk=args.topk, use_memory=use_memory)
            streams = SyntheticStreams(task, args.streams, seed=seed)
            name = f"synth_{'cave' if use_memory else 'sans_cave'}_s{seed}"
            log = train_streams(
                model, streams, eval_batches, CAT_NAMES, steps=args.steps, lr=args.lr,
                warmup=100, val_every=max(args.steps // 8, 1), mem_size=args.mem_size,
                vram_slots=args.vram_slots, device=device, run_dir=args.run_dir, run_name=name,
                extra_config={'task': 'synthetic_recall', 'seed': seed},
            )
            results[(seed, use_memory)] = log['final']['val']

    chance = 1 / task.n_values
    print(f"\n{'=' * 72}\nPrecision sur les valeurs a retrouver (hasard = {100 * chance:.1f}%)\n{'=' * 72}")
    print(f"{'seed':>6} | {'proche sans/avec cave':>22} | {'loin sans/avec cave':>22} | VRAM caviste vs LRU")
    rows = []
    for seed in args.seeds:
        a, b = results[(seed, False)], results[(seed, True)]
        cv = b.get('caviste', {})
        pct = lambda e: f"{100 * e['acc']:5.1f}%" if e['n'] else '   n/a'
        print(f"{seed:>6} | {pct(a['rappel_proche']):>10} {pct(b['rappel_proche']):>10} | "
              f"{pct(a['rappel_loin']):>10} {pct(b['rappel_loin']):>10} | "
              f"{100 * cv.get('caviste_hit_rate', 0):.1f}% vs {100 * cv.get('lru_hit_rate', 0):.1f}%")
        rows.append({'seed': seed, 'sans_cave': a, 'avec_cave': b})
    with open(os.path.join(args.run_dir, 'synthetic_recall_summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
