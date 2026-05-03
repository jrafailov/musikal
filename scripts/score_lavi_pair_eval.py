"""
Score Lavi's filled-in pair ratings against the answer key. Reports the
same accuracy metric the models report (binary pair classification on
the held-out test split), so her bar can sit next to the model bars on
the poster chart.

Reads:
  data/clean/baseline/lavi_pair_eval_template.csv   filled with y/n in 'rating'
  data/clean/baseline/lavi_pair_eval_answer_key.csv

Writes:
  data/clean/baseline/lavi_pair_eval_results.json   accuracy + confusion + per-class
  prints a comparison table against the model accuracies in model_metrics.txt
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "clean" / "baseline"

TEMPLATE = OUT_DIR / "lavi_pair_eval_template.csv"
ANSWER_KEY = OUT_DIR / "lavi_pair_eval_answer_key.csv"
RESULTS_OUT = OUT_DIR / "lavi_pair_eval_results.json"
METRICS_TXT = OUT_DIR / "model_metrics.txt"


def parse_rating(v):
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s in {"y", "yes", "1", "true", "mix", "mixable"}:
        return 1
    if s in {"n", "no", "0", "false", "nomix", "not"}:
        return 0
    return None


ratings = pd.read_csv(TEMPLATE)
answers = pd.read_csv(ANSWER_KEY)

merged = ratings.merge(answers, on="pair_id", how="inner")
merged["pred"] = merged["rating"].apply(parse_rating)

unrated = merged[merged.pred.isna()]
if len(unrated) > 0:
    print(f"WARNING: {len(unrated)} pairs have unparseable ratings (skipping):")
    print(unrated[["pair_id", "rating"]].to_string(index=False))

scored = merged.dropna(subset=["pred"]).copy()
scored["pred"] = scored["pred"].astype(int)

y_true = scored["true_label"].values
y_pred = scored["pred"].values

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Wilson 95% CI for accuracy so we can show error bars on the poster
n = len(y_true)
p = acc
z = 1.96
denom = 1 + z**2 / n
center = (p + z**2 / (2 * n)) / denom
half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
ci_lo, ci_hi = center - half, center + half

print()
print("=" * 60)
print(f"LAVI HUMAN BASELINE  ({n} rated pairs)")
print("=" * 60)
print(f"  Accuracy:  {acc:.3f}   (95% CI {ci_lo:.3f} – {ci_hi:.3f})")
print(f"  Precision: {prec:.3f}")
print(f"  Recall:    {rec:.3f}")
print(f"  F1:        {f1:.3f}")
print()
print("  Confusion matrix:")
print(f"                pred neg    pred pos")
print(f"    actual neg  {cm[0,0]:8d}    {cm[0,1]:8d}")
print(f"    actual pos  {cm[1,0]:8d}    {cm[1,1]:8d}")

# show next to model accuracies if they're available
print()
print("=" * 60)
print("COMPARISON TO MODELS")
print("=" * 60)
if METRICS_TXT.exists():
    print(METRICS_TXT.read_text())
print(f"  Lavi (human):         {acc:.3f}")

results = {
    "n_rated": int(n),
    "n_skipped": int(len(unrated)),
    "accuracy": float(acc),
    "accuracy_ci_lo": float(ci_lo),
    "accuracy_ci_hi": float(ci_hi),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "confusion_matrix": {
        "true_neg": int(cm[0, 0]),
        "false_pos": int(cm[0, 1]),
        "false_neg": int(cm[1, 0]),
        "true_pos": int(cm[1, 1]),
    },
}
RESULTS_OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved {RESULTS_OUT}")
