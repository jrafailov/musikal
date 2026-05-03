"""
Sample a balanced 30-pair slice from the held-out test split for Lavi to
rate manually. Apples-to-apples with the model pair-classification metric:
same pairs.parquet, same split, same 50/50 positive/negative balance.

The shuffle seed is grid-searched so prefix balance stays close to 50/50
at n=10, 15, 20. This way an early stop still gives a usable accuracy
estimate without the prefix being skewed.

Outputs three files in data/clean/baseline/:
  lavi_pair_eval_questions.md     blind questionnaire with YT links
  lavi_pair_eval_template.csv     pair_id + blank column for her ratings
  lavi_pair_eval_answer_key.csv   true labels (kept on our side, gitignored)

Each pair shows two YouTube links and asks whether they would mix into
each other. No info about whether it's a real DJ transition or random.
"""

from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIRS = REPO_ROOT / "data" / "clean" / "baseline" / "pairs.parquet"
TRACKS = REPO_ROOT / "data" / "clean" / "tracks.csv"
OUT_DIR = REPO_ROOT / "data" / "clean" / "baseline"

QUESTIONS_OUT = OUT_DIR / "lavi_pair_eval_questions.md"
TEMPLATE_OUT = OUT_DIR / "lavi_pair_eval_template.csv"
ANSWER_KEY_OUT = OUT_DIR / "lavi_pair_eval_answer_key.csv"

SAMPLE_SEED = 11
N_POS = 15
N_NEG = 15
PREFIX_CHECKPOINTS = (10, 15, 20)
SHUFFLE_SEED_CANDIDATES = range(500)


def yt(tid):
    return f"https://youtube.com/watch?v={tid}"


def fmt(tid, tracks):
    if tid in tracks.index:
        row = tracks.loc[tid]
        artist = row.get("artist") if isinstance(row.get("artist"), str) else "?"
        title = row.get("title") if isinstance(row.get("title"), str) else "?"
        return f"{artist} — {title}"
    return f"[{tid}]"


pairs = pd.read_parquet(PAIRS)
tracks = pd.read_csv(TRACKS).set_index("track_id")

test = pairs[pairs.split == "test"].reset_index(drop=True)
test_pos = test[test.label == 1]
test_neg = test[test.label == 0]

print(f"Test pool: {len(test_pos)} positives, {len(test_neg)} negatives")

pos_sample = test_pos.sample(n=N_POS, random_state=SAMPLE_SEED)
neg_sample = test_neg.sample(n=N_NEG, random_state=SAMPLE_SEED)
combined = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)


def prefix_imbalance(labels, checkpoints):
    """Sum of absolute deviations from 50/50 balance at each prefix length."""
    total = 0
    for k in checkpoints:
        pos_k = int((labels[:k] == 1).sum())
        total += abs(pos_k - k / 2)
    return total


best_seed = None
best_score = float("inf")
for seed in SHUFFLE_SEED_CANDIDATES:
    shuffled = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    score = prefix_imbalance(shuffled["label"].values, PREFIX_CHECKPOINTS)
    if score < best_score:
        best_score = score
        best_seed = seed
        if score == 0:
            break

print(f"Picked shuffle seed {best_seed} (prefix imbalance score {best_score})")
sample = combined.sample(frac=1, random_state=best_seed).reset_index(drop=True)
sample["pair_id"] = np.arange(1, len(sample) + 1)

print("Prefix balance check:")
for k in list(PREFIX_CHECKPOINTS) + [len(sample)]:
    pos_k = int((sample["label"].values[:k] == 1).sum())
    neg_k = k - pos_k
    print(f"  first {k:2d}: {pos_k} pos / {neg_k} neg")

# blind questionnaire
lines = []
lines.append("# Lavi Pair Evaluation — Mixable or Not")
lines.append("")
lines.append(f"{len(sample)} track pairs from our held-out test set, balanced 50/50 between "
             "real DJ transitions and random pairs (you don't know which is which).")
lines.append("")
lines.append("**Task:** for each pair, listen to both tracks and decide whether they would "
             "mix well into each other in a DJ set. Mark `y` (would mix) or `n` (would not) "
             "in the second column of `lavi_pair_eval_template.csv`.")
lines.append("")
lines.append("Order is randomized. Don't try to guess our side, just rate each one on its "
             "own merits.")
lines.append("")

for _, row in sample.iterrows():
    pid = int(row.pair_id)
    a, b = row.track_1, row.track_2
    lines.append(f"## Pair {pid}")
    lines.append("")
    lines.append(f"**Track A:** [{fmt(a, tracks)}]({yt(a)})")
    lines.append("")
    lines.append(f"**Track B:** [{fmt(b, tracks)}]({yt(b)})")
    lines.append("")

QUESTIONS_OUT.write_text("\n".join(lines))

# template she fills in
template = pd.DataFrame({
    "pair_id": sample.pair_id,
    "rating": "",
})
template.to_csv(TEMPLATE_OUT, index=False)

# answer key for scoring
answer_key = pd.DataFrame({
    "pair_id": sample.pair_id,
    "track_1": sample.track_1,
    "track_2": sample.track_2,
    "true_label": sample.label,
})
answer_key.to_csv(ANSWER_KEY_OUT, index=False)

print(f"\nWrote {QUESTIONS_OUT}")
print(f"Wrote {TEMPLATE_OUT}")
print(f"Wrote {ANSWER_KEY_OUT}")
print(f"  {len(sample)} pairs total ({(sample.label == 1).sum()} pos, "
      f"{(sample.label == 0).sum()} neg)")
