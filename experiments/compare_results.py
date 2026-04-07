"""Compare les resultats de toutes les experiences."""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS_DIR = "D:/Nature/runs"


def load_run(run_name):
    """Charge le log d'un run."""
    log_path = os.path.join(RUNS_DIR, run_name, "log.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        return json.load(f)


def get_best_val(log):
    """Extrait la meilleure val loss d'un log."""
    if not log or 'steps' not in log:
        return None
    val_losses = [s['val_loss'] for s in log['steps'] if 'val_loss' in s]
    return min(val_losses) if val_losses else None


def get_best_step(log):
    """Extrait le step de la meilleure val loss."""
    if not log or 'steps' not in log:
        return None
    best_val = get_best_val(log)
    for s in log['steps']:
        if s.get('val_loss') == best_val:
            return s['step']
    return None


def compare_experiments(experiments):
    """
    experiments: dict de {nom: [run_names]}
    """
    print(f"\n{'='*70}")
    print("COMPARAISON DES EXPERIENCES")
    print(f"{'='*70}\n")

    results = {}
    for exp_name, run_names in experiments.items():
        vals = []
        for rn in run_names:
            log = load_run(rn)
            if log is None:
                print(f"  [SKIP] {rn} - pas de log.json")
                continue
            bv = get_best_val(log)
            bs = get_best_step(log)
            if bv is not None:
                vals.append(bv)
                print(f"  {rn}: best_val={bv:.4f} (step {bs})")

        if vals:
            vals = np.array(vals)
            results[exp_name] = {
                'mean': vals.mean(),
                'std': vals.std(),
                'min': vals.min(),
                'max': vals.max(),
                'n_seeds': len(vals),
            }

    if len(results) < 2:
        print("\nPas assez d'experiences terminees pour comparer.")
        return results

    print(f"\n{'='*70}")
    print(f"{'Experience':<25} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Seeds':>5}")
    print(f"{'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}")
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean']):
        print(f"{name:<25} | {r['mean']:8.4f} | {r['std']:8.4f} | {r['min']:8.4f} | {r['max']:8.4f} | {r['n_seeds']:5d}")

    # Comparaisons pair-a-pair
    names = list(results.keys())
    if len(names) >= 2:
        print(f"\n{'='*70}")
        print("DELTAS (negatif = amelioration)")
        print(f"{'='*70}")
        baseline_name = [n for n in names if 'baseline' in n.lower() and '6l' not in n.lower() and '8l' not in n.lower()]
        if not baseline_name:
            baseline_name = [names[0]]
        baseline_name = baseline_name[0]
        bl = results[baseline_name]

        for name, r in results.items():
            if name == baseline_name:
                continue
            delta = r['mean'] - bl['mean']
            pct = delta / bl['mean'] * 100
            print(f"  {name} vs {baseline_name}: {delta:+.4f} ({pct:+.1f}%)")

    return results


if __name__ == "__main__":
    # Detecter les runs disponibles
    if not os.path.exists(RUNS_DIR):
        print("Pas de runs dir")
        sys.exit(1)

    all_runs = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]

    # Grouper par experience
    experiments = {}
    for run in all_runs:
        if run.startswith("baseline_full_"):
            experiments.setdefault("baseline_6L", []).append(run)
        elif run.startswith("baseline_8L_"):
            experiments.setdefault("baseline_8L", []).append(run)
        elif run.startswith("afm_"):
            experiments.setdefault("AFMT", []).append(run)

    for name, runs in experiments.items():
        print(f"{name}: {runs}")

    compare_experiments(experiments)
