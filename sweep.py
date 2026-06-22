#!/usr/bin/env python3
"""
Hyperparameter sweep for lca_sim_mldc_SingleSnaptshot.py.

Searches stride × kernel_size × features, runs experiments in parallel
(one per GPU), parses best-compression metrics from each run.log, and
produces a summary table + Pareto plot (rel_err vs comp_ratio).

Usage
-----
    python sweep.py                        # 4 GPUs, default config
    python sweep.py --gpus 2               # 2 GPUs
    python sweep.py --config my.yaml       # different base config
    python sweep.py --features 64 128      # include 128-feature runs
    python sweep.py --quick                # short training (faster sweep)
"""

import argparse
import glob
import itertools
import os
import re
import subprocess
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------

DEFAULT_GRID = {
    'stride':      [4, 6, 8],
    'kernel_size': [7, 9, 11],
    'features':    [64],        # extend via --features flag
}

# Reduced training budget used during sweep (overrides base config values)
SWEEP_OVERRIDES = {
    'max_epochs':         100,
    'lambda_anneal_stop':  30,
    'stabilize_epochs':     8,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_valid(stride, kernel_size, patch_size):
    return (
        kernel_size % 2 == 1        # lcapt requires odd kernel
        and kernel_size >= stride   # no receptive-field gaps at inference
        and patch_size % stride == 0
    )


def parse_log(log_path):
    """
    Return the best-compression metrics line from run.log:
      [best] CompRatio=X  λ=Y  rel_err=Z  BPV=W
    Falls back to last epoch's values if no [best] line exists.
    """
    result = dict(comp_ratio=None, lambda_=None, rel_err=None, bpv=None)
    if not os.path.exists(log_path):
        return result
    with open(log_path) as f:
        content = f.read()

    best_matches = re.findall(
        r'\[best\] CompRatio=([0-9.]+)x\s+λ=([0-9.]+)\s+rel_err=([0-9.]+)\s+BPV=([0-9.]+)',
        content,
    )
    if best_matches:
        m = best_matches[-1]
        result.update(comp_ratio=float(m[0]), lambda_=float(m[1]),
                      rel_err=float(m[2]), bpv=float(m[3]))
        return result

    # Fallback: last epoch line
    epoch_matches = re.findall(
        r'CompRatio:\s*([0-9.]+)x\s+BPV:\s*([0-9.]+)',
        content,
    )
    rel_err_matches = re.findall(r'Rel\.err:\s*([0-9.]+)', content)
    lambda_matches  = re.findall(r'λ=([0-9.]+)', content)
    if epoch_matches:
        result['comp_ratio'] = float(epoch_matches[-1][0])
        result['bpv']        = float(epoch_matches[-1][1])
    if rel_err_matches:
        result['rel_err']  = float(rel_err_matches[-1])
    if lambda_matches:
        result['lambda_']  = float(lambda_matches[-1])
    return result


def find_exp_dir(sweep_id):
    """Locate the experiment directory whose copied config contains sweep_id."""
    for d in sorted(glob.glob('experiments/simmldc_*'), reverse=True):
        cfg_path = os.path.join(d, 'config_simmldc.yaml')
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        if cfg.get('_sweep_id') == sweep_id:
            return d
    return None


def pareto_front(points):
    """Return indices of Pareto-optimal points (min rel_err, max comp_ratio)."""
    front = []
    for i, (e, c) in enumerate(points):
        dominated = False
        for j, (ej, cj) in enumerate(points):
            if i != j and ej <= e and cj >= c and (ej < e or cj > c):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus',     type=int, default=4)
    parser.add_argument('--config',   default='config_simmldc.yaml')
    parser.add_argument('--features', type=int, nargs='+', default=None,
                        help='feature counts to sweep (default: [64])')
    parser.add_argument('--quick',    action='store_true',
                        help='halve the training budget for faster iteration')
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    patch_size = base_cfg['data']['patch_size']
    features   = args.features or DEFAULT_GRID['features']

    overrides = dict(SWEEP_OVERRIDES)
    if args.quick:
        overrides['max_epochs']         = 50
        overrides['lambda_anneal_stop'] = 15

    # Build valid combos
    combos = []
    for stride, kernel, feat in itertools.product(
        DEFAULT_GRID['stride'], DEFAULT_GRID['kernel_size'], features
    ):
        if is_valid(stride, kernel, patch_size):
            combos.append(dict(stride=stride, kernel_size=kernel, features=feat))

    print(f"Patch size : {patch_size}  |  {len(combos)} valid combinations\n")
    for c in combos:
        code_pos = c['features'] * (patch_size // c['stride'])**3
        print(f"  stride={c['stride']}  kernel={c['kernel_size']}  "
              f"features={c['features']}  code_pos/patch={code_pos:,}")

    os.makedirs('sweep_configs', exist_ok=True)

    jobs = []
    for i, combo in enumerate(combos):
        cfg = deepcopy(base_cfg)
        cfg['model']['stride']      = combo['stride']
        cfg['model']['kernel_size'] = combo['kernel_size']
        cfg['model']['features']    = combo['features']
        for k, v in overrides.items():
            cfg['training'][k] = v

        sweep_id = (f"sweep_{i:03d}"
                    f"_s{combo['stride']}"
                    f"_k{combo['kernel_size']}"
                    f"_f{combo['features']}")
        cfg['_sweep_id'] = sweep_id

        cfg_path = os.path.join('sweep_configs', f'{sweep_id}.yaml')
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        jobs.append(dict(sweep_id=sweep_id, combo=combo, cfg_path=cfg_path))

    # Run in batches of --gpus experiments (one per GPU)
    all_results = {}
    for batch_start in range(0, len(jobs), args.gpus):
        batch = jobs[batch_start:batch_start + args.gpus]
        print(f"\n--- Batch {batch_start // args.gpus + 1} "
              f"({len(batch)} jobs) ---")

        procs = []
        for gpu_id, job in enumerate(batch):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            cmd = ['python', 'lca_sim_mldc_SingleSnaptshot.py', job['cfg_path']]
            print(f"  GPU {gpu_id}: {job['sweep_id']}")
            p = subprocess.Popen(cmd, env=env,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            procs.append((job, p))

        for job, p in procs:
            p.wait()
            exp_dir = find_exp_dir(job['sweep_id'])
            metrics = parse_log(os.path.join(exp_dir, 'run.log')) if exp_dir else {}
            all_results[job['sweep_id']] = dict(combo=job['combo'],
                                                exp_dir=exp_dir,
                                                **metrics)
            status = ('OK' if metrics.get('comp_ratio') else 'NO METRICS')
            print(f"  {job['sweep_id']}  [{status}]"
                  + (f"  comp={metrics['comp_ratio']:.2f}x"
                     f"  rel_err={metrics['rel_err']:.4f}"
                     if metrics.get('comp_ratio') else ''))

    # Summary table
    print('\n' + '=' * 90)
    print('SWEEP RESULTS  (sorted by compression ratio, best-compression checkpoint)')
    print('=' * 90)
    print(f"{'stride':>6} {'kernel':>6} {'feat':>6} | "
          f"{'CompRatio':>9} {'BPV':>6} {'rel_err':>8} {'λ':>6} | exp_dir")
    print('-' * 90)

    sorted_res = sorted(all_results.items(),
                        key=lambda x: x[1].get('comp_ratio') or 0,
                        reverse=True)
    for sid, r in sorted_res:
        c    = r['combo']
        comp = r.get('comp_ratio')
        bpv  = r.get('bpv')
        err  = r.get('rel_err')
        lam  = r.get('lambda_')
        edir = r.get('exp_dir') or 'N/A'
        print(f"{c['stride']:>6} {c['kernel_size']:>6} {c['features']:>6} | "
              f"{comp or 'N/A':>9} {bpv or 'N/A':>6} {err or 'N/A':>8} "
              f"{lam or 'N/A':>6} | {os.path.basename(edir)}")

    # Pareto plot
    valid = [(sid, r) for sid, r in all_results.items()
             if r.get('comp_ratio') and r.get('rel_err')]
    if valid:
        errs   = [r['rel_err']   for _, r in valid]
        comps  = [r['comp_ratio'] for _, r in valid]
        labels = [f"s{r['combo']['stride']}/k{r['combo']['kernel_size']}/f{r['combo']['features']}"
                  for _, r in valid]

        front_idx = pareto_front(list(zip(errs, comps)))

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(errs, comps, zorder=3, s=80, color='steelblue')

        for i, (e, c, lbl) in enumerate(zip(errs, comps, labels)):
            ax.annotate(lbl, (e, c), textcoords='offset points',
                        xytext=(5, 4), fontsize=8)
            if i in front_idx:
                ax.scatter(e, c, zorder=4, s=120, edgecolors='red',
                           facecolors='none', linewidths=1.5)

        # Connect Pareto front
        front_pts = sorted([(errs[i], comps[i]) for i in front_idx])
        if len(front_pts) > 1:
            fx, fy = zip(*front_pts)
            ax.step(fx, fy, where='post', color='red', linewidth=1,
                    linestyle='--', label='Pareto front')
            ax.legend(fontsize=9)

        ax.axvline(0.01, color='orange', linestyle=':', linewidth=1,
                   label='1% error target')
        ax.set_xlabel('Relative Error (lower is better)', fontsize=11)
        ax.set_ylabel('Compression Ratio (higher is better)', fontsize=11)
        ax.set_title(f'Hyperparameter Sweep — patch_size={patch_size}', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = 'sweep_pareto.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nPareto plot saved to {plot_path}")

    results_path = 'sweep_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print(f"Full results saved to {results_path}")


if __name__ == '__main__':
    main()
