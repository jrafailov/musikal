"""
Download YouTube audio for every mir-aidj track that appears in enough mixes.

Pre-filters the raw djmix-dataset.json with a Counter to keep only track_ids
that appear in at least `--min-mixes` distinct mixes (default 3 — takes the
corpus from 63K down to ~4.1K and the runtime from days to ~12-24 hours).

For each kept track, pulls mp3 audio via yt-dlp into data/audio/{track_id}.mp3.
Resumable: skips any track_id whose mp3 already exists. Failures are appended
to data/audio-log/failures.csv as (timestamp, track_id, reason) so a 10-20%
failure rate from taken-down, geo-blocked, or age-restricted videos doesn't
kill the run.

Legal note: academic research context. Audio stays internal for feature
extraction, never redistributed. mp3s are gitignored; sync via DVC.

Usage
    python scripts/download_audio.py
    python scripts/download_audio.py --min-mixes 3 --limit 5     # smoke test
"""

import argparse
import csv
import json
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yt_dlp

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = REPO_ROOT / "data" / "mir-aidj-raw" / "djmix-dataset.json"
AUDIO_DIR = REPO_ROOT / "data" / "audio"
LOG_DIR = REPO_ROOT / "data" / "audio-log"
FAILURES_CSV = LOG_DIR / "failures.csv"

FAILURE_HEADER = ["ts_utc", "track_id", "reason"]
INTERRUPTED = False


def filter_track_ids(raw_json_path, min_mixes):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mix_counts = Counter()
    for mix in data:
        ids_in_this_mix = {e.get("id") for e in mix.get("tracklist", []) if e.get("id")}
        for tid in ids_in_this_mix:
            mix_counts[tid] += 1

    kept = sorted(tid for tid, c in mix_counts.items() if c >= min_mixes)
    return kept, len(mix_counts)


def append_failure(track_id, reason):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not FAILURES_CSV.exists()
    with open(FAILURES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FAILURE_HEADER)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        w.writerow([ts, track_id, reason[:500]])


def trim_reason(err):
    """yt-dlp errors are verbose. Keep the first line which is usually the
    actionable bit (e.g., 'Video unavailable', 'Private video', 'Sign in to
    confirm you're not a bot')."""
    msg = str(err).strip().splitlines()
    return msg[0] if msg else "unknown"


def build_ydl_opts():
    return {
        "format": "bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"},
        ],
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
    }


def handle_sigint(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("\n[interrupt] finishing current track then exiting cleanly...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mixes", type=int, default=3,
                    help="keep track_ids appearing in >= this many distinct mixes")
    ap.add_argument("--throttle-secs", type=float, default=2.5,
                    help="sleep between requests to avoid YouTube rate limiting")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N kept track_ids (smoke test)")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    kept, total_ids = filter_track_ids(RAW_JSON, args.min_mixes)
    filter_count = len(kept)
    if args.limit:
        kept = kept[:args.limit]

    already = {p.stem for p in AUDIO_DIR.glob("*.mp3")}
    todo = [tid for tid in kept if tid not in already]

    print(f"unique track_ids in raw:  {total_ids}")
    print(f"after >= {args.min_mixes} mixes filter: {filter_count}"
          + (f" (limited to first {len(kept)})" if args.limit else ""))
    print(f"already downloaded:       {len(already & set(kept))}")
    print(f"todo this run:            {len(todo)}")
    print(f"throttle:                 {args.throttle_secs}s between requests")
    print()

    signal.signal(signal.SIGINT, handle_sigint)

    ydl_opts = build_ydl_opts()
    succeeded = 0
    failed = 0
    start_ts = time.monotonic()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, track_id in enumerate(todo, start=1):
            if INTERRUPTED:
                break

            url = f"https://youtube.com/watch?v={track_id}"
            try:
                ydl.download([url])
                succeeded += 1
            except yt_dlp.utils.DownloadError as e:
                append_failure(track_id, trim_reason(e))
                failed += 1
            except Exception as e:
                append_failure(track_id, f"unexpected: {trim_reason(e)}")
                failed += 1

            if i % 100 == 0 or i == len(todo):
                elapsed = time.monotonic() - start_ts
                rate = i / max(elapsed, 1e-6)
                eta_min = (len(todo) - i) / max(rate, 1e-6) / 60
                print(f"[{i}/{len(todo)}] ok={succeeded} fail={failed} "
                      f"rate={rate:.2f}/s eta={eta_min:.0f}min", flush=True)

            time.sleep(args.throttle_secs)

    print()
    print(f"done. succeeded={succeeded} failed={failed} "
          f"(of {len(todo)} attempted this run)")
    if failed:
        print(f"failures logged to {FAILURES_CSV}")
    if INTERRUPTED:
        sys.exit(130)


if __name__ == "__main__":
    main()
