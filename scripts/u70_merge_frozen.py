#!/usr/bin/env python3
"""U.7.0a: Merge Rust frozen OOW into consolidated results, recompute CIs."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y_true, y_score, n=1000, seed=42):
    """Row-resampled bootstrap CI on ROC-AUC."""
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true, dtype=float)
    y_score = np.array(y_score, dtype=float)
    aucs = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2 or len(np.unique(ys)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, ys)))
    if not aucs:
        return None, [None, None]
    return float(np.mean(aucs)), [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


# Load frozen eval set
with open('data/oow_eval_set.json') as f:
    frozen = json.load(f)
frozen_hashes = {}
for name, entries in frozen['repos'].items():
    frozen_hashes[name] = set(e['hash'] for e in entries)
    print(f'Frozen {name}: {len(entries)} commits available')

# Load u69 scores for django, react, kafka, kubernetes
results = {}
for name in ['django', 'react', 'kafka', 'kubernetes']:
    u69 = json.load(open(f'data/u69_{name}_oow.json'))
    scores = u69['scores']
    fh = frozen_hashes.get(name, set())
    matched = [s for s in scores if s['hash'] in fh]
    print(f'\n{name}: {len(matched)}/{len(scores)} u69 overlap with frozen ({len(fh)} total)')
    
    if len(matched) < 10:
        print(f'  WARNING: too few overlaps, using all u69 scores')
        matched = scores
    
    y_true = [s['actual'] for s in matched]
    y_score = [s['score'] for s in matched]
    n = len(y_true)
    rate = sum(y_true) / n if n else 0
    uniq = len(set(round(s, 6) for s in y_score))
    
    if uniq < 2 or sum(y_true) == 0 or sum(y_true) == n:
        print(f'  DEGENERATE: {uniq} scores, {sum(y_true)}/{n} pos')
        results[name] = {'n': n, 'rate': rate, 'auc': None, 'mean': None, 'ci': [None, None]}
        continue
    
    auc = float(roc_auc_score(np.array(y_true), np.array(y_score)))
    mean, ci = bootstrap_auc(y_true, y_score)
    results[name] = {'n': n, 'rate': rate, 'auc': auc, 'mean': mean, 'ci': ci}
    print(f'  AUC={auc:.4f}, mean={mean:.4f}, CI=[{ci[0]:.4f}, {ci[1]:.4f}]')

# Load Rust from the v2 frozen scoring (N=50)
print('\n--- Rust (from u70_rust_frozen_v2, N=50) ---')
with open('data/u70_rust_frozen_oow.json') as f:
    rust_frozen = json.load(f)
rust_scores = rust_frozen['scores']
y_true_r = [s['actual'] for s in rust_scores]
y_score_r = [s['score'] for s in rust_scores]
n_r = len(y_true_r)
rate_r = sum(y_true_r) / n_r if n_r else 0
uniq_r = len(set(round(s, 6) for s in y_score_r))
print(f'  N={n_r}, base_rate={rate_r:.3f}, unique_scores={uniq_r}, pos={sum(y_true_r)}')

if uniq_r >= 2 and 0 < sum(y_true_r) < n_r:
    auc_r = float(roc_auc_score(np.array(y_true_r), np.array(y_score_r)))
    mean_r, ci_r = bootstrap_auc(y_true_r, y_score_r)
    results['rust'] = {'n': n_r, 'rate': rate_r, 'auc': auc_r, 'mean': mean_r, 'ci': ci_r}
    print(f'  AUC={auc_r:.4f}, mean={mean_r:.4f}, CI=[{ci_r[0]:.4f}, {ci_r[1]:.4f}]')
else:
    print(f'  DEGENERATE: cannot compute AUC')
    results['rust'] = {'n': n_r, 'rate': rate_r, 'auc': None, 'mean': None, 'ci': [None, None]}

# Print summary
print('\n' + '='*65)
print('FROZEN OOW HEADLINE (final, committed to data/u70_frozen_oow.json)')
print('='*65)
print(f'{"Repo":12s} {"N":>5s} {"Rate":>7s} {"AUC":>7s} {"CI lo":>7s} {"CI hi":>7s}')
print('-' * 50)
aucs = []
for name in ['django', 'react', 'kafka', 'kubernetes', 'rust']:
    r = results.get(name, {})
    if r.get('auc') is not None:
        aucs.append(r['auc'])
        print(f'{name:12s} {r["n"]:5d} {r["rate"]:7.3f} {r["auc"]:7.4f} {r["ci"][0]:7.4f} {r["ci"][1]:7.4f}')
    else:
        print(f'{name:12s} {r.get("n","?"):>5s} {r.get("rate","?"):>7s} {"UNDEF":>7s}')
if aucs:
    print(f'{"MEAN":12s} {"":>5s} {"":>7s} {np.mean(aucs):7.4f}')
print()

# Save final
with open('data/u70_frozen_oow.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to data/u70_frozen_oow.json')
