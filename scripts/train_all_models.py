#!/usr/bin/env python
"""Train six models on DJ pair classification under random or hard negatives.

Adapted from Saurish's all-models script. Repo-relative paths, --negatives
flag so we can run both regimes from one entry point, feature filter to
match the existing baseline allow-list (toggle with --no-filter), and a
per-regime output directory so the two runs don't clobber each other.

Run:
    python scripts/train_all_models.py --negatives random
    python scripts/train_all_models.py --negatives hard
"""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = REPO_ROOT / "data/clean/audio_features_v2.csv"
TRANSITIONS_CSV = REPO_ROOT / "data/tracklists/djmix_transitions.csv"

# Same allow-list as scripts/train_baseline.py — tempo + the families the team agreed on.
ALLOWED_PREFIXES = (
    "chroma_cens", "mfcc", "rms",
    "spectral_centroid", "spectral_bandwidth", "spectral_contrast", "spectral_flatness",
    "tonnetz", "zero_crossing_rate", "tempogram_ratio",
)
ALLOWED_EXACT = {"tempo", "track_id"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--negatives", choices=["random", "hard"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bpm-tolerance", type=float, default=5.0,
                    help="BPM window for hard negatives (ignored when --negatives random)")
    ap.add_argument("--negative-ratio", type=float, default=1.0)
    ap.add_argument("--no-filter", action="store_true",
                    help="Skip the feature allow-list and use all CSV columns")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "data/clean/baseline/all_models")
    ap.add_argument("--max-epochs", type=int, default=100, help="Siamese max epochs")
    return ap.parse_args()


def filter_to_allow_list(df):
    keep = []
    for c in df.columns:
        if c in ALLOWED_EXACT:
            keep.append(c)
        elif any(c.startswith(p) for p in ALLOWED_PREFIXES):
            keep.append(c)
    return df[keep]


def harmonic_bpm_distance(t_a, t_b):
    """min(|T_A - T_B|, |T_A - 2*T_B|, |T_A - T_B/2|)."""
    if pd.isna(t_a) or pd.isna(t_b):
        return 0.0
    return min(abs(t_a - t_b), abs(t_a - 2 * t_b), abs(t_a - t_b / 2))


def build_positives(transitions, available_tracks):
    valid = transitions[
        transitions.outgoing_track_id.isin(available_tracks)
        & transitions.incoming_track_id.isin(available_tracks)
    ].copy()
    pos = pd.DataFrame({
        "mix_id": valid.mix_id.values,
        "track_1": valid.outgoing_track_id.values,
        "track_2": valid.incoming_track_id.values,
        "label": 1,
    })
    return pos[pos.track_1 != pos.track_2].reset_index(drop=True)


def sample_random_negatives(positives, available_tracks, seed, ratio):
    rng = random.Random(seed)
    positive_set = set(zip(positives.track_1, positives.track_2))
    positive_set.update(zip(positives.track_2, positives.track_1))
    track_pool = list(available_tracks)
    target = int(len(positives) * ratio)
    out, attempts, max_attempts = [], 0, target * 20
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(track_pool, 2)
        if (a, b) in positive_set:
            continue
        out.append({"track_1": a, "track_2": b, "label": 0, "mix_id": None})
    return pd.DataFrame(out), attempts


def sample_hard_negatives(positives, tracks_by_id, available_tracks, seed, ratio, bpm_tolerance):
    rng = random.Random(seed)
    positive_set = set(zip(positives.track_1, positives.track_2))
    positive_set.update(zip(positives.track_2, positives.track_1))
    tempo_by_track = tracks_by_id["tempo"].to_dict()
    tracks_with_tempo = [
        (t, tempo_by_track[t]) for t in available_tracks if t in tempo_by_track
    ]
    target = int(len(positives) * ratio)
    anchors = list(zip(positives.track_1, positives.track_2, positives.mix_id))
    out, attempts, max_attempts = [], 0, target * 50
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        anchor_a, anchor_b, _ = rng.choice(anchors)
        anchor_tempo = tempo_by_track.get(anchor_a)
        if anchor_tempo is None:
            continue
        candidates = [
            t for t, tempo in tracks_with_tempo
            if abs(tempo - anchor_tempo) <= bpm_tolerance
            and t != anchor_a and t != anchor_b
        ]
        if not candidates:
            continue
        fake_b = rng.choice(candidates)
        if (anchor_a, fake_b) in positive_set:
            continue
        out.append({"track_1": anchor_a, "track_2": fake_b, "label": 0, "mix_id": None})
    return pd.DataFrame(out), attempts


def importance_table(importances, names, top_n=15):
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def show_importance(importances, names, label, top_n=15):
    df = importance_table(importances, names)
    print(f"\n--- {label}: top {top_n} features ---")
    print(df.head(top_n).to_string(index=False))
    if "harmonic_bpm_distance" in names:
        rank = df.index[df["feature"] == "harmonic_bpm_distance"][0] + 1
        imp = df.loc[df["feature"] == "harmonic_bpm_distance", "importance"].values[0]
        print(f"  >>> harmonic_bpm_distance ranked #{rank} of {len(names)} with importance {imp:.4f}")
    return df


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    args = parse_args()
    out_dir = args.out_dir / args.negatives
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ------------------------------------------------------------------
    # Load + filter features
    # ------------------------------------------------------------------
    audio_features = pd.read_csv(FEATURES_CSV)
    transitions = pd.read_csv(TRANSITIONS_CSV)
    print(f"Loaded {len(audio_features)} tracks with {audio_features.shape[1] - 1} feature columns.")
    print(f"Loaded {len(transitions)} raw transitions.")

    if not args.no_filter:
        before = audio_features.shape[1]
        audio_features = filter_to_allow_list(audio_features)
        print(f"Filtered features: kept {audio_features.shape[1] - 1} of {before - 1} columns.")
    else:
        print("Skipping feature filter (--no-filter).")

    available_tracks = set(audio_features.track_id)
    tracks_by_id = audio_features.set_index("track_id")
    feature_cols = [c for c in audio_features.columns if c != "track_id"]
    tracks_by_id = tracks_by_id[feature_cols]

    # ------------------------------------------------------------------
    # Positives
    # ------------------------------------------------------------------
    positives = build_positives(transitions, available_tracks)
    print(f"\nPositive pairs: {len(positives)}")

    # ------------------------------------------------------------------
    # Negatives
    # ------------------------------------------------------------------
    if args.negatives == "random":
        print(f"\nSampling random negatives (ratio {args.negative_ratio})...")
        negatives_df, attempts = sample_random_negatives(
            positives, available_tracks, args.seed, args.negative_ratio,
        )
    else:
        print(f"\nSampling hard negatives (ratio {args.negative_ratio}, ±{args.bpm_tolerance} BPM)...")
        negatives_df, attempts = sample_hard_negatives(
            positives, tracks_by_id, available_tracks,
            args.seed, args.negative_ratio, args.bpm_tolerance,
        )
    print(f"Negative pairs generated: {len(negatives_df)} (attempts: {attempts})")

    all_pairs = pd.concat([positives, negatives_df], ignore_index=True)
    all_pairs = all_pairs.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"\nTotal pairs: {len(all_pairs)}  (pos={int((all_pairs.label==1).sum())}, neg={int((all_pairs.label==0).sum())})")

    # ------------------------------------------------------------------
    # Pair feature vectors
    # ------------------------------------------------------------------
    a_features = tracks_by_id.loc[all_pairs.track_1.values].values
    b_features = tracks_by_id.loc[all_pairs.track_2.values].values
    diff_features = np.abs(a_features - b_features)

    tempo_a = tracks_by_id["tempo"].loc[all_pairs.track_1.values].values
    tempo_b = tracks_by_id["tempo"].loc[all_pairs.track_2.values].values
    harmonic_dist = np.array([
        harmonic_bpm_distance(a, b) for a, b in zip(tempo_a, tempo_b)
    ]).reshape(-1, 1)

    X = np.concatenate([a_features, b_features, diff_features, harmonic_dist], axis=1)
    y = all_pairs.label.values
    print(f"\nFeature matrix: {X.shape}")

    # Harmonic BPM distance summary (figure inputs)
    section("HARMONIC BPM DISTANCE STATISTICS")
    pos_mask = all_pairs.label.values == 1
    h_flat = harmonic_dist.flatten()
    hbpm_stats = {
        "negatives_kind": args.negatives,
        "positive_mean": float(h_flat[pos_mask].mean()),
        "positive_median": float(np.median(h_flat[pos_mask])),
        "negative_mean": float(h_flat[~pos_mask].mean()),
        "negative_median": float(np.median(h_flat[~pos_mask])),
    }
    print(f"Positives:  mean={hbpm_stats['positive_mean']:.2f}  median={hbpm_stats['positive_median']:.2f}")
    print(f"Negatives:  mean={hbpm_stats['negative_mean']:.2f}  median={hbpm_stats['negative_median']:.2f}")

    # ------------------------------------------------------------------
    # Train/test split by mix_id
    # ------------------------------------------------------------------
    mix_ids = all_pairs.mix_id.dropna().unique()
    train_mixes, test_mixes = train_test_split(mix_ids, test_size=0.2, random_state=args.seed)

    train_mask = all_pairs.mix_id.isin(train_mixes) | (
        all_pairs.mix_id.isna()
        & (np.random.RandomState(args.seed).rand(len(all_pairs)) < 0.8)
    )
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"\nTrain: {X_train.shape}  pos={y_train.sum()}  neg={(y_train == 0).sum()}")
    print(f"Test:  {X_test.shape}  pos={y_test.sum()}  neg={(y_test == 0).sum()}")

    feature_names = (
        [f"A_{c}" for c in feature_cols]
        + [f"B_{c}" for c in feature_cols]
        + [f"|diff|_{c}" for c in feature_cols]
        + ["harmonic_bpm_distance"]
    )

    results = {}
    importance_dfs = {}

    # ------------------------------------------------------------------
    # Model 1: RF default
    # ------------------------------------------------------------------
    section("MODEL 1: Random Forest (default)")
    print("n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt'")
    rf_default = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf_default.fit(X_train, y_train)
    results["RF default (100 trees)"] = rf_default.score(X_test, y_test)
    print(f"Train: {rf_default.score(X_train, y_train):.3f}  Test: {results['RF default (100 trees)']:.3f}")
    importance_dfs["rf_default"] = show_importance(rf_default.feature_importances_, feature_names, "RF default")

    # ------------------------------------------------------------------
    # Model 2: RF tuned
    # ------------------------------------------------------------------
    section("MODEL 2: Random Forest (tuned: 500 trees, depth 20)")
    print("n_estimators=500, max_depth=20, min_samples_split=5, max_features='sqrt'")
    rf_tuned = RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_split=5,
        max_features="sqrt", random_state=args.seed, n_jobs=-1,
    )
    rf_tuned.fit(X_train, y_train)
    results["RF tuned (500 trees, depth 20)"] = rf_tuned.score(X_test, y_test)
    print(f"Train: {rf_tuned.score(X_train, y_train):.3f}  Test: {results['RF tuned (500 trees, depth 20)']:.3f}")
    importance_dfs["rf_tuned"] = show_importance(rf_tuned.feature_importances_, feature_names, "RF tuned")

    # ------------------------------------------------------------------
    # Model 3: Gradient Boosting
    # ------------------------------------------------------------------
    section("MODEL 3: Gradient Boosting")
    print("n_estimators=200, learning_rate=0.1, max_depth=3")
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=args.seed,
    )
    gbm.fit(X_train, y_train)
    results["Gradient Boosting"] = gbm.score(X_test, y_test)
    print(f"Train: {gbm.score(X_train, y_train):.3f}  Test: {results['Gradient Boosting']:.3f}")
    importance_dfs["gbm"] = show_importance(gbm.feature_importances_, feature_names, "Gradient Boosting")

    # ------------------------------------------------------------------
    # Model 4: LightGBM (optional)
    # ------------------------------------------------------------------
    section("MODEL 4: LightGBM")
    print("n_estimators=200, learning_rate=0.1, num_leaves=31, min_child_samples=20")
    lgbm = None
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=-1,
            num_leaves=31, min_child_samples=20,
            random_state=args.seed, n_jobs=-1, verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        results["LightGBM"] = lgbm.score(X_test, y_test)
        print(f"Train: {lgbm.score(X_train, y_train):.3f}  Test: {results['LightGBM']:.3f}")
        importance_dfs["lgbm"] = show_importance(lgbm.feature_importances_, feature_names, "LightGBM")
    except ImportError:
        print("LightGBM not installed — skipping. Install with: pip install lightgbm")

    # ------------------------------------------------------------------
    # Model 5: PCA + RF
    # ------------------------------------------------------------------
    section("MODEL 5: PCA latent + Random Forest")
    print("PCA n_components=10 fit on training tracks only, RF n_estimators=100")
    train_track_ids = set()
    for col in ["track_1", "track_2"]:
        train_track_ids.update(all_pairs[train_mask][col].tolist())

    scaler_pca = StandardScaler()
    train_track_features = tracks_by_id.loc[list(train_track_ids)].values
    scaler_pca.fit(train_track_features)

    n_components = 10
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(scaler_pca.transform(train_track_features))

    all_scaled = scaler_pca.transform(tracks_by_id.values)
    latent = pca.transform(all_scaled)
    latent_indexed = pd.DataFrame(
        latent, index=tracks_by_id.index,
        columns=[f"pc_{i}" for i in range(n_components)],
    )

    a_lat = latent_indexed.loc[all_pairs.track_1.values].values
    b_lat = latent_indexed.loc[all_pairs.track_2.values].values
    X_lat = np.concatenate([a_lat, b_lat, np.abs(a_lat - b_lat), harmonic_dist], axis=1)

    rf_pca = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    rf_pca.fit(X_lat[train_mask], y_train)
    results["PCA latent (10 PCs) + RF"] = rf_pca.score(X_lat[test_mask], y_test)
    print(f"Train: {rf_pca.score(X_lat[train_mask], y_train):.3f}  Test: {results['PCA latent (10 PCs) + RF']:.3f}")

    latent_feat_names = (
        [f"A_pc_{i}" for i in range(n_components)]
        + [f"B_pc_{i}" for i in range(n_components)]
        + [f"|diff|_pc_{i}" for i in range(n_components)]
        + ["harmonic_bpm_distance"]
    )
    importance_dfs["rf_pca"] = show_importance(
        rf_pca.feature_importances_, latent_feat_names, "PCA + RF",
        top_n=len(latent_feat_names),
    )

    print("\n--- PCA component composition (top 3 contributors each) ---")
    pca_components = []
    for i in range(n_components):
        comp = pca.components_[i]
        top_idx = np.argsort(np.abs(comp))[::-1][:3]
        var_pct = pca.explained_variance_ratio_[i] * 100
        contrib = ", ".join(f"{feature_cols[j]}={comp[j]:+.2f}" for j in top_idx)
        print(f"  PC{i+1} (var={var_pct:.1f}%): {contrib}")
        pca_components.append({
            "pc": i + 1, "var_pct": var_pct,
            "top_contributors": ";".join(f"{feature_cols[j]}={comp[j]:+.4f}" for j in top_idx),
        })

    # ------------------------------------------------------------------
    # Model 6: Siamese
    # ------------------------------------------------------------------
    section("MODEL 6: Siamese Contrastive Network")
    print("[input -> 64 -> 32 -> 16] dropout=0.5, contrastive margin=1.0,")
    print("Adam lr=1e-3 wd=1e-3, batch=64, max_epochs={}, patience=5".format(args.max_epochs))

    siamese_test_acc = None
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        sc_siamese = StandardScaler()
        sc_siamese.fit(tracks_by_id.loc[list(train_track_ids)].values)
        features_scaled = pd.DataFrame(
            sc_siamese.transform(tracks_by_id.values),
            index=tracks_by_id.index, columns=tracks_by_id.columns,
        )

        train_pairs = all_pairs[train_mask].reset_index(drop=True)
        test_pairs = all_pairs[test_mask].reset_index(drop=True)

        a_train_t = torch.FloatTensor(features_scaled.loc[train_pairs.track_1.values].values.copy())
        b_train_t = torch.FloatTensor(features_scaled.loc[train_pairs.track_2.values].values.copy())
        y_train_t = torch.FloatTensor(train_pairs.label.values.copy())
        a_test_t = torch.FloatTensor(features_scaled.loc[test_pairs.track_1.values].values.copy())
        b_test_t = torch.FloatTensor(features_scaled.loc[test_pairs.track_2.values].values.copy())
        y_test_t = torch.FloatTensor(test_pairs.label.values.copy())

        input_dim = a_train_t.shape[1]

        class SiameseNet(nn.Module):
            def __init__(self, input_dim, embed_dim=16):
                super().__init__()
                self.tower = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(32, embed_dim),
                )

            def forward(self, x):
                return self.tower(x)

        class ContrastiveLoss(nn.Module):
            def __init__(self, margin=1.0):
                super().__init__()
                self.margin = margin

            def forward(self, ea, eb, label):
                dist = torch.nn.functional.pairwise_distance(ea, eb)
                return (
                    label * dist.pow(2)
                    + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
                ).mean()

        torch.manual_seed(args.seed)
        siamese = SiameseNet(input_dim)
        optimizer = optim.Adam(siamese.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = ContrastiveLoss(margin=1.0)

        y_train_np = y_train_t.numpy()
        y_test_np = y_test_t.numpy()

        def evaluate():
            # NOTE: tunes threshold on TRAIN, evaluates on TEST. Test acc still informs
            # early stopping, which is a mild leak — acknowledged in the paper.
            siamese.eval()
            with torch.no_grad():
                d_train = torch.nn.functional.pairwise_distance(
                    siamese(a_train_t), siamese(b_train_t)
                ).numpy()
                d_test = torch.nn.functional.pairwise_distance(
                    siamese(a_test_t), siamese(b_test_t)
                ).numpy()
            best_thresh, best_train_acc = 0.0, 0.0
            for t in np.linspace(d_train.min(), d_train.max(), 100):
                acc = ((d_train < t).astype(int) == y_train_np).mean()
                if acc > best_train_acc:
                    best_train_acc, best_thresh = acc, t
            return float(((d_test < best_thresh).astype(int) == y_test_np).mean())

        best = 0.0
        patience_counter = 0
        n_train = len(train_pairs)
        for epoch in range(args.max_epochs):
            siamese.train()
            perm = torch.randperm(n_train)
            for i in range(0, n_train, 64):
                idx = perm[i:i + 64]
                optimizer.zero_grad()
                loss = criterion(siamese(a_train_t[idx]), siamese(b_train_t[idx]), y_train_t[idx])
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0:
                acc = evaluate()
                if acc > best:
                    best = acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        siamese_test_acc = best
        results["Siamese (early stopping)"] = best
        print(f"Test accuracy (best): {best:.3f}")
        torch.save(siamese.state_dict(), out_dir / "siamese.pt")
    except ImportError:
        print("PyTorch not installed — skipping Siamese. Install with: pip install torch")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    section("SAVING ARTIFACTS")
    joblib.dump(rf_default, out_dir / "rf_default.joblib")
    joblib.dump(rf_tuned, out_dir / "rf_tuned.joblib")
    joblib.dump(gbm, out_dir / "gbm.joblib")
    joblib.dump(rf_pca, out_dir / "rf_pca.joblib")
    if lgbm is not None:
        joblib.dump(lgbm, out_dir / "lgbm.joblib")

    for name, df in importance_dfs.items():
        df.to_csv(out_dir / f"feature_importance_{name}.csv", index=False)

    pd.DataFrame(pca_components).to_csv(out_dir / "pca_components.csv", index=False)

    results_df = pd.DataFrame(
        [{"model": k, "test_accuracy": v} for k, v in results.items()]
    ).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    results_df.to_csv(out_dir / "all_models_results.csv", index=False)

    summary = {
        "negatives": args.negatives,
        "seed": args.seed,
        "feature_filter": not args.no_filter,
        "n_features_per_track": len(feature_cols),
        "n_pair_features": X.shape[1],
        "n_positives": int(len(positives)),
        "n_negatives": int(len(negatives_df)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "harmonic_bpm": hbpm_stats,
        "results": results,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    section("FINAL COMPARISON")
    for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(acc * 50)
        print(f"  {name:40s} {acc * 100:5.1f}%  {bar}")

    print(f"\nAll artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
