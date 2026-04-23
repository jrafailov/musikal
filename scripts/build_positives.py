"""
Walk each mir-aidj tracklist and emit positive transition pairs.

For every mix, sort its tracks by position_in_mix and emit a row for every
consecutive pair (track_a_id, track_b_id, mix_id). We keep mix_id on every
row because the train/val split is done at the mix level, not the pair level.

Inputs
    data/tracklists/djmix_tracks.csv    walked tracks per mix (mix_id, track_id, position_in_mix, ...)
    data/clean/tracks.csv               deduped valid track_ids (Jasmine's output)

Output
    data/clean/positives.parquet        columns: track_a_id, track_b_id, mix_id

Design notes
    1. Missing-track handling. Jasmine's clean/tracks.csv drops tracks whose
       raw title couldn't be parsed into "artist - title" (~14% of tracklist
       rows). When a dropped track sits in the middle of a mix [A, B, *C*, D]
       we DROP AROUND the gap: emit (A,B), skip (B,C) and (C,D). We do not
       bridge to synthesize (B,D). Rationale: a positive should be a real
       observed DJ transition, not an inferred one. Bridging would let
       unrelated tracks from either side of an unparseable title get labeled
       as a good transition.

    2. Self-loops. We drop pairs where track_a_id == track_b_id (~23 rows).
       These are almost certainly annotation artifacts (same YouTube ID listed
       back-to-back).
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKS_IN_MIXES = REPO_ROOT / "data" / "tracklists" / "djmix_tracks.csv"
CLEAN_TRACKS = REPO_ROOT / "data" / "clean" / "tracks.csv"
OUT_PATH = REPO_ROOT / "data" / "clean" / "positives.parquet"


def walk_transitions(tracks_in_mixes):
    """
    Given the flat per-mix track list, emit consecutive (a, b) pairs within
    each mix, ordered by position_in_mix.
    """
    tracks_in_mixes = tracks_in_mixes.sort_values(["mix_id", "position_in_mix"])

    a = tracks_in_mixes.rename(columns={"track_id": "track_a_id"})
    a["track_b_id"] = a.groupby("mix_id")["track_a_id"].shift(-1)

    pairs = a.dropna(subset=["track_b_id"])[["track_a_id", "track_b_id", "mix_id"]]
    return pairs.reset_index(drop=True)


def main():
    tracks_in_mixes = pd.read_csv(TRACKS_IN_MIXES, usecols=["mix_id", "track_id", "position_in_mix"])
    clean = pd.read_csv(CLEAN_TRACKS, usecols=["track_id"])

    print(f"tracklist rows: {len(tracks_in_mixes)}")
    print(f"clean track_ids: {len(clean)}")

    pairs = walk_transitions(tracks_in_mixes)
    print(f"raw consecutive pairs: {len(pairs)}")

    valid_ids = set(clean["track_id"])
    keep = pairs["track_a_id"].isin(valid_ids) & pairs["track_b_id"].isin(valid_ids)
    positives = pairs[keep].reset_index(drop=True)
    dropped_clean = len(pairs) - len(positives)

    self_loops = positives["track_a_id"] == positives["track_b_id"]
    positives = positives[~self_loops].reset_index(drop=True)

    print(f"dropped (endpoint not in clean set): {dropped_clean}")
    print(f"dropped (self-loop a==b):             {self_loops.sum()}")
    print(f"kept positives: {len(positives)}")
    print(f"mixes represented: {positives['mix_id'].nunique()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    positives.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
