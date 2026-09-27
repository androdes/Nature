"""
Memoire vivante sur le vrai corpus : la cave aide-t-elle la ou elle doit aider ?

Chaque document (fichier du corpus) est lu dans l'ordre, segment par
segment. On compare, SEED PAR SEED (memes donnees, meme init), le meme
modele avec et sans cave, et on decoupe la loss de validation par
categorie de token (src/stream_data.py) :

  nouveau    trigramme jamais vu plus tot dans le document
  proche     deja vu dans le segment courant (l'attention locale suffit)
  loin       deja vu plus tot, hors du segment, mais dans la cave
  hors_cave  deja vu, mais plus ancien que la cave

Prediction si la cave marche : gain net sur "loin", rien sur "nouveau".
Un gain uniforme sur toutes les categories signalerait au contraire un
simple effet de capacite (comme SimpleMem).

    python experiments/run_memoire_vivante.py                        # 3 seeds x 2 conditions
    python experiments/run_memoire_vivante.py --seeds 42 --steps 500 # verification rapide
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import DataConfig, TransformerConfig
from src.memorizing_transformer import MemorizingTransformer
from src.stream_data import (load_documents, DocumentStreams, build_eval_batches,
                             token_categories, CAT_NAMES)
from src.stream_train import train_streams


def main():
    d = DataConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 137, 256])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--streams', type=int, default=32)
    ap.add_argument('--seq-len', type=int, default=256)
    ap.add_argument('--mem-size', type=int, default=16384, help='fiches par flux (tokens de passe)')
    ap.add_argument('--vram-slots', type=int, default=2048, help='capacite VRAM simulee du caviste')
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--mem-layer', type=int, default=None)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--val-every', type=int, default=250)
    ap.add_argument('--max-eval-segments', type=int, default=None)
    ap.add_argument('--data-dir', default=d.data_dir)
    ap.add_argument('--tokenizer', default=d.tokenizer_path)
    ap.add_argument('--run-dir', default='D:/Nature/runs')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_cfg = DataConfig(data_dir=args.data_dir, tokenizer_path=args.tokenizer, seq_len=args.seq_len)
    train_docs, val_docs, tokenizer = load_documents(data_cfg)
    eval_batches = build_eval_batches(
        val_docs, args.streams, args.seq_len,
        cats_fn=lambda doc: token_categories(doc, args.seq_len, args.mem_size),
        max_segments=args.max_eval_segments,
    )
    counts = {name: int(sum(int(((b[2] == c) & (b[1] != -100)).sum()) for b in eval_batches))
              for c, name in CAT_NAMES.items()}
    print(f"Tokens de validation par categorie : {counts}")

    cfg = TransformerConfig(vocab_size=tokenizer.get_vocab_size(), max_seq_len=args.seq_len)
    results = {}
    for seed in args.seeds:
        for use_memory in (False, True):
            torch.manual_seed(seed)
            model = MemorizingTransformer(cfg, mem_layer=args.mem_layer, topk=args.topk,
                                          use_memory=use_memory)
            streams = DocumentStreams(train_docs, args.streams, args.seq_len, seed=seed)
            name = f"vivante_{'cave' if use_memory else 'sans_cave'}_m{args.mem_size}_s{seed}"
            log = train_streams(
                model, streams, eval_batches, CAT_NAMES, steps=args.steps, lr=args.lr,
                val_every=args.val_every, mem_size=args.mem_size, vram_slots=args.vram_slots,
                device=device, run_dir=args.run_dir, run_name=name,
                extra_config={'task': 'corpus', 'seed': seed, 'streams': args.streams},
            )
            results[(seed, use_memory)] = log['best']['val']

    # Comparaison appariee (meme seed = memes donnees, meme init)
    print(f"\n{'=' * 78}\nMeilleure val loss, apparie par seed (avec cave - sans cave ; negatif = mieux)\n{'=' * 78}")
    names = ['loss'] + list(CAT_NAMES.values())
    print(f"{'seed':>6} | " + ' | '.join(f"{n:>10}" for n in names))
    deltas = {n: [] for n in names}
    rows = []
    for seed in args.seeds:
        a, b = results[(seed, False)], results[(seed, True)]
        line = []
        for n in names:
            va = a['loss'] if n == 'loss' else a[n]['loss']
            vb = b['loss'] if n == 'loss' else b[n]['loss']
            dlt = None if va is None or vb is None else vb - va
            if dlt is not None:
                deltas[n].append(dlt)
            line.append(f"{dlt:+10.4f}" if dlt is not None else f"{'n/a':>10}")
        print(f"{seed:>6} | " + ' | '.join(line))
        rows.append({'seed': seed, 'sans_cave': a, 'avec_cave': b})
    mean = [f"{sum(v) / len(v):+10.4f}" if v else f"{'n/a':>10}" for v in deltas.values()]
    print(f"{'moy':>6} | " + ' | '.join(mean))
    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, f'memoire_vivante_m{args.mem_size}_summary.json'), 'w') as f:
        json.dump(rows, f, indent=2)


if __name__ == '__main__':
    main()
