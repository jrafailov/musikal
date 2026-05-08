# musikal

Supervised next track recommender for DJs. NYU Courant ML, Spring 2026.

writeup: [paper/main.pdf](paper/main.pdf) · [poster/poster.pdf](poster/poster.pdf)

## Setup

```bash
git clone git@github.com:jrafailov/musikal.git
cd musikal
pip install -r requirements.txt
```

Python 3.10+. 
training inputs are already in `data/clean/`. 
raw audio is local-only.

## Run

essential scripts are in [scripts/](scripts/):

```bash
python scripts/train_baseline.py           ### baseline RF + GBM sweep
python scripts/train_all_models.py         ### full model comparison
python scripts/retrieval_eval.py           ### recall@k eval
python scripts/recommend_for_audio.py path/to/track.mp3
```

Feature extraction (`extract_features_v2.py`, `extract_features_v3.py`) needs raw audio and is run on JR's station only.
Notebooks for exploratory analysis are in [notebooks/](notebooks/).

