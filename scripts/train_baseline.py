from pathlib import Path

import pandas as pd
import numpy as np
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKLISTS_DIR = REPO_ROOT / "data" / "tracklists"
CLEAN_DIR = REPO_ROOT / "data" / "clean"
OUT_DIR = CLEAN_DIR / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================
dj_mix = pd.read_csv(TRACKLISTS_DIR / "djmix_mixes.csv")
dj_tracks = pd.read_csv(TRACKLISTS_DIR / "djmix_tracks.csv")
dj_transitions = pd.read_csv(TRACKLISTS_DIR / "djmix_transitions.csv")

# ============================================================
# EDA: Shapes and columns
# ============================================================
print("=" * 60)
print("SHAPES")
print("=" * 60)
print(f"Mixes:       {dj_mix.shape}")
print(f"Tracks:      {dj_tracks.shape}")
print(f"Transitions: {dj_transitions.shape}")

print("=" * 60)
print("COLUMNS")
print("=" * 60)
print("Mixes:      ", list(dj_mix.columns))
print("Tracks:     ", list(dj_tracks.columns))
print("Transitions:", list(dj_transitions.columns))

# ============================================================
# EDA: Track universe
# ============================================================
print("=" * 60)
print("TRACK UNIVERSE")
print("=" * 60)
unique_tracks = dj_tracks.track_id.dropna().nunique()
total_appearances = dj_tracks.track_id.notna().sum()
print(f"Unique tracks:       {unique_tracks}")
print(f"Total appearances:   {total_appearances}")
print(f"Avg appearances/track: {total_appearances / unique_tracks:.2f}")

appearances_per_track = dj_tracks.track_id.value_counts()
print(f"\nDistribution of appearances per track:")
print(appearances_per_track.describe())
print(f"\nTop 10 most-played tracks:")
print(appearances_per_track.head(10))

# ============================================================
# EDA: Coverage analysis
# ============================================================
print("=" * 60)
print("COVERAGE BY APPEARANCE COUNT")
print("=" * 60)
appearances = dj_tracks.track_id.value_counts()
for top_n in [1000, 5000, 10000, 20000, 30000]:
    top_tracks = set(appearances.head(top_n).index)
    covered = dj_transitions[
        dj_transitions.outgoing_track_id.isin(top_tracks) &
        dj_transitions.incoming_track_id.isin(top_tracks)
    ]
    print(f"Top {top_n} tracks cover {len(covered)} transitions ({len(covered) / len(dj_transitions) * 100:.1f}%)")

print("=" * 60)
print("COVERAGE BY TRANSITION PARTICIPATION")
print("=" * 60)
transition_counts = pd.concat([
    dj_transitions.outgoing_track_id,
    dj_transitions.incoming_track_id
]).value_counts()

print("Tracks participating in most transitions:")
print(transition_counts.head(20))

for top_n in [5000, 10000, 20000, 30000]:
    top_tracks = set(transition_counts.head(top_n).index)
    covered = dj_transitions[
        dj_transitions.outgoing_track_id.isin(top_tracks) &
        dj_transitions.incoming_track_id.isin(top_tracks)
    ]
    print(f"Top {top_n} by transition-count cover {len(covered)} transitions ({len(covered) / len(dj_transitions) * 100:.1f}%)")

# ============================================================
# Load audio features (v2)
# ============================================================
audio_features = pd.read_csv(CLEAN_DIR / "audio_features_v2.csv")
print(f"\nLoaded v2 audio features: {len(audio_features)} tracks, {audio_features.shape[1] - 1} feature columns")

# ============================================================
# Filter to ONLY the features the team specified (allow-list)
# ============================================================
ALLOWED_PREFIXES = (
    "chroma_cens",
    "mfcc",
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_contrast",
    "spectral_flatness",
    "tonnetz",
    "zero_crossing_rate",
    "tempogram_ratio",
)
ALLOWED_EXACT = {"tempo", "track_id"}

cols_to_keep = []
for c in audio_features.columns:
    if c in ALLOWED_EXACT:
        cols_to_keep.append(c)
        continue
    if any(c.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        cols_to_keep.append(c)

print(f"\nKeeping {len(cols_to_keep) - 1} feature columns (plus track_id).")
print(f"Dropped {audio_features.shape[1] - len(cols_to_keep)} columns.")

audio_features = audio_features[cols_to_keep]

# Show what families survived
remaining_families = {}
for c in audio_features.columns:
    if c == "track_id":
        continue
    if c == "tempo":
        family = "tempo"
    elif c.startswith("tempogram_ratio"):
        family = "tempogram_ratio"
    elif c.startswith("chroma_cens"):
        family = "chroma_cens"
    elif c.startswith("mfcc"):
        family = "mfcc"
    elif c.startswith("spectral_contrast"):
        family = "spectral_contrast"
    elif c.startswith("spectral_centroid"):
        family = "spectral_centroid"
    elif c.startswith("spectral_bandwidth"):
        family = "spectral_bandwidth"
    elif c.startswith("spectral_flatness"):
        family = "spectral_flatness"
    elif c.startswith("zero_crossing_rate"):
        family = "zero_crossing_rate"
    elif c.startswith("rms"):
        family = "rms"
    elif c.startswith("tonnetz"):
        family = "tonnetz"
    else:
        family = f"unmatched ({c})"
    remaining_families[family] = remaining_families.get(family, 0) + 1

print("Remaining feature families and their column counts:")
for fam, count in sorted(remaining_families.items(), key=lambda x: -x[1]):
    print(f"  {fam}: {count}")

track_features_with_id = set(audio_features.track_id)

# ============================================================
# Generate positive pairs
# ============================================================
transitions = dj_transitions[
    dj_transitions.outgoing_track_id.isin(track_features_with_id) &
    dj_transitions.incoming_track_id.isin(track_features_with_id)
].copy()

positives = pd.DataFrame({
    "mix_id": transitions.mix_id.values,
    "track_1": transitions.outgoing_track_id.values,
    "track_2": transitions.incoming_track_id.values,
    "label": 1,
})
positives = positives[positives.track_1 != positives.track_2].reset_index(drop=True)
print(f"\nPositive pairs: {len(positives)}")
print(positives.head())

# ============================================================
# Generate RANDOM negative pairs
# ============================================================
RANDOM_SEED = 42
NEGATIVE_RATIO = 1.0
rng = random.Random(RANDOM_SEED)

positive_set = set(zip(positives.track_1, positives.track_2))
positive_set.update(zip(positives.track_2, positives.track_1))

all_tracks = list(track_features_with_id)
n_negatives = int(len(positives) * NEGATIVE_RATIO)

negatives = []
attempts = 0
max_attempts = n_negatives * 20

while len(negatives) < n_negatives and attempts < max_attempts:
    attempts += 1
    a, b = rng.sample(all_tracks, 2)
    if (a, b) in positive_set:
        continue
    negatives.append({
        "track_1": a,
        "track_2": b,
        "label": 0,
        "mix_id": None,
    })

negatives_df = pd.DataFrame(negatives)
print(f"Negative pairs: {len(negatives_df)}")

# ============================================================
# Combine and shuffle pairs
# ============================================================
all_pairs = pd.concat([positives, negatives_df], ignore_index=True)
all_pairs = all_pairs.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\nTotal pairs: {len(all_pairs)}")
print(f"Positives: {(all_pairs.label == 1).sum()}")
print(f"Negatives: {(all_pairs.label == 0).sum()}")

# ============================================================
# Build pair feature vectors
# ============================================================
features_indexed = audio_features.set_index("track_id")
feature_cols = [c for c in audio_features.columns if c != "track_id"]
features_indexed = features_indexed[feature_cols]

a_features = features_indexed.loc[all_pairs.track_1.values].values
b_features = features_indexed.loc[all_pairs.track_2.values].values
diff_features = np.abs(a_features - b_features)

X = np.concatenate([a_features, b_features, diff_features], axis=1)
y = all_pairs.label.values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels: positives={y.sum()}, negatives={(y == 0).sum()}")

# ============================================================
# Train/test split by mix_id
# ============================================================
mix_ids = all_pairs.mix_id.dropna().unique()
train_mixes, test_mixes = train_test_split(mix_ids, test_size=0.2, random_state=42)

train_mask = all_pairs.mix_id.isin(train_mixes) | (
    all_pairs.mix_id.isna() & (np.random.RandomState(42).rand(len(all_pairs)) < 0.8)
)
test_mask = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"\nTrain: {X_train.shape}, positives: {y_train.sum()}, negatives: {(y_train == 0).sum()}")
print(f"Test:  {X_test.shape}, positives: {y_test.sum()}, negatives: {(y_test == 0).sum()}")

# ============================================================
# BASELINE: default random forest (this should give the ~75% number)
# ============================================================
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
print("\nTraining default random forest (baseline)...")
rf.fit(X_train, y_train)
print("Done.")

train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Train-test gap: {train_acc - test_acc:.3f}")

# Confusion matrix and per-class breakdown on the test set
y_pred_baseline = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_baseline)
print("\nConfusion matrix (default RF baseline, test set):")
print(f"                pred neg    pred pos")
print(f"  actual neg  {cm[0,0]:8d}    {cm[0,1]:8d}")
print(f"  actual pos  {cm[1,0]:8d}    {cm[1,1]:8d}")
print()
print(classification_report(
    y_test,
    y_pred_baseline,
    target_names=["random_pair", "real_transition"],
    digits=3,
))

# ============================================================
# Per-column feature importance
# ============================================================
feature_names = (
    [f"A_{c}" for c in feature_cols] +
    [f"B_{c}" for c in feature_cols] +
    [f"|diff|_{c}" for c in feature_cols]
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 features by importance (per column):")
print(importance_df.head(15).to_string(index=False))

# ============================================================
# Per-base-feature importance (aggregated across A/B/|diff|)
# ============================================================
def base_feature_name(prefixed):
    """Strip A_, B_, |diff|_ prefix to get the underlying feature name."""
    for prefix in ["A_", "B_", "|diff|_"]:
        if prefixed.startswith(prefix):
            return prefixed[len(prefix):]
    return prefixed

base_importance = {}
for name, imp in zip(feature_names, rf.feature_importances_):
    base = base_feature_name(name)
    base_importance[base] = base_importance.get(base, 0) + imp

base_importance_df = pd.DataFrame([
    {"feature": k, "importance": v} for k, v in base_importance.items()
]).sort_values("importance", ascending=False)

print("\nTop 15 base features (aggregated across A, B, and |diff|):")
print(base_importance_df.head(15).to_string(index=False))

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)

tuning_results = []

# Try several RF configurations
rf_configs = [
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "max_features": "sqrt"},  # baseline
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 20, "min_samples_split": 5, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 10, "min_samples_split": 10, "max_features": "sqrt"},
    {"n_estimators": 1000, "max_depth": 15, "min_samples_split": 5, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "max_features": 0.3},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "max_features": 0.5},
]

print("\n--- Random Forest configurations ---")
best_rf_test = test_acc
best_rf_config = "default (n=100, depth=None, split=2)"

# Track the best model object across all configs and both classes,
# starting from the default RF baseline.
best_overall_test = test_acc
best_overall_model = rf
best_overall_config = "default RF (n=100, depth=None, split=2)"
best_overall_class = "RandomForest"

for config in rf_configs:
    rf_t = RandomForestClassifier(random_state=42, n_jobs=-1, **config)
    rf_t.fit(X_train, y_train)
    train_t = rf_t.score(X_train, y_train)
    test_t = rf_t.score(X_test, y_test)

    config_str = f"n={config['n_estimators']}, depth={config['max_depth']}, split={config['min_samples_split']}, feat={config['max_features']}"
    print(f"  {config_str}: train={train_t:.3f}, test={test_t:.3f}")

    tuning_results.append({
        "model": "RandomForest",
        "config": config_str,
        "train_acc": train_t,
        "test_acc": test_t,
    })

    if test_t > best_rf_test:
        best_rf_test = test_t
        best_rf_config = config_str
    if test_t > best_overall_test:
        best_overall_test = test_t
        best_overall_model = rf_t
        best_overall_config = config_str
        best_overall_class = "RandomForest"

# Try gradient boosting
print("\n--- Gradient Boosting ---")
gbm_configs = [
    {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 7},
]

best_gbm_test = 0
best_gbm_config = ""

for config in gbm_configs:
    gbm = GradientBoostingClassifier(random_state=42, **config)
    gbm.fit(X_train, y_train)
    train_g = gbm.score(X_train, y_train)
    test_g = gbm.score(X_test, y_test)

    config_str = f"n={config['n_estimators']}, lr={config['learning_rate']}, depth={config['max_depth']}"
    print(f"  {config_str}: train={train_g:.3f}, test={test_g:.3f}")

    tuning_results.append({
        "model": "GradientBoosting",
        "config": config_str,
        "train_acc": train_g,
        "test_acc": test_g,
    })

    if test_g > best_gbm_test:
        best_gbm_test = test_g
        best_gbm_config = config_str
    if test_g > best_overall_test:
        best_overall_test = test_g
        best_overall_model = gbm
        best_overall_config = config_str
        best_overall_class = "GradientBoosting"

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Default RF baseline:  {test_acc:.3f}")
print(f"  Best tuned RF:        {best_rf_test:.3f}  ({best_rf_test - test_acc:+.3f})  [{best_rf_config}]")
print(f"  Best gradient boost:  {best_gbm_test:.3f}  ({best_gbm_test - test_acc:+.3f})  [{best_gbm_config}]")

best_overall = max(test_acc, best_rf_test, best_gbm_test)
print(f"\nOverall best test accuracy: {best_overall:.3f}")

# ============================================================
# Save outputs
# ============================================================
import json

joblib.dump(rf, OUT_DIR / "rf_model_baseline.joblib")
joblib.dump(best_overall_model, OUT_DIR / "best_model.joblib")
importance_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
base_importance_df.to_csv(OUT_DIR / "feature_importance_base.csv", index=False)
pd.DataFrame(tuning_results).to_csv(OUT_DIR / "tuning_results.csv", index=False)

# Persist the train/test split and the exact feature column order so
# downstream scripts (retrieval_eval.py, ablation, etc.) can load and
# reuse instead of rebuilding pairs from scratch with different RNG impls.
pairs_with_split = all_pairs.copy()
pairs_with_split["split"] = np.where(train_mask, "train", "test")
pairs_with_split.to_parquet(OUT_DIR / "pairs.parquet", index=False)

with open(OUT_DIR / "feature_cols.json", "w") as f:
    json.dump({
        "feature_cols": feature_cols,
        "best_class": best_overall_class,
        "best_config": best_overall_config,
        "best_test_acc": float(best_overall_test),
    }, f, indent=2)

with open(OUT_DIR / "model_metrics.txt", "w") as f:
    f.write(f"Allowed feature families: {ALLOWED_PREFIXES + tuple(ALLOWED_EXACT)}\n")
    f.write(f"Negative sampling: random\n")
    f.write(f"Total pairs: {len(all_pairs)}\n")
    f.write(f"Train pairs: {len(y_train)}\n")
    f.write(f"Test pairs:  {len(y_test)}\n")
    f.write(f"Positives: {(all_pairs.label == 1).sum()}\n")
    f.write(f"Negatives: {(all_pairs.label == 0).sum()}\n")
    f.write(f"\n--- Default RF baseline ({X_train.shape[1]} features) ---\n")
    f.write(f"Train accuracy: {train_acc:.3f}\n")
    f.write(f"Test accuracy:  {test_acc:.3f}\n")
    f.write(f"\n--- Hyperparameter tuning results ---\n")
    for r in tuning_results:
        f.write(f"  {r['model']}: {r['config']}: test={r['test_acc']:.3f}\n")
    f.write(f"\n--- Final summary ---\n")
    f.write(f"Default RF:           {test_acc:.3f}\n")
    f.write(f"Best tuned RF:        {best_rf_test:.3f} [{best_rf_config}]\n")
    f.write(f"Best gradient boost:  {best_gbm_test:.3f} [{best_gbm_config}]\n")
    f.write(f"Overall best:         {best_overall:.3f}\n")

print("\nSaved baseline model, feature importance, tuning results, and metrics.")
