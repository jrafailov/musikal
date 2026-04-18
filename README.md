# musikal
Project for NYU Courant's machine learning course. A supervised learning model that recommends the next track for DJs based on audio features.

## Getting started

### 1. Clone the repo

```bash
git clone git@github.com:jrafailov/musikal.git
cd musikal
```

### 2. Install Python dependencies

```bash
pip install dvc dvc-gdrive
```

You'll also want the project dependencies (librosa, pandas, etc.) but DVC is what you need for the data.

### 3. Pull the data from Google Drive

The `data/` directory is tracked with [DVC](https://dvc.org/) and stored in a shared Google Drive folder. It's about 10GB, so make sure you have the space and a decent connection.

```bash
dvc pull
```

This will prompt you to authenticate with Google the first time. A browser window will open asking you to sign in. Use the Google account that has access to the shared Drive folder.

If `dvc pull` gives you permission errors, make sure you've been added to the shared folder:

https://drive.google.com/drive/u/1/folders/1QrJ94QLWWPc1GcTbtcjFf6bpu6WOL5v_

Ask JR to share access if you don't have it.

### 4. Verify the data

After the pull finishes, you should see the `data/` directory populated:

```bash
ls data/
```

If something looks wrong or incomplete, try:

```bash
dvc status
```

This tells you if your local data matches what's tracked.

### Updating data

If new data gets added to the project:

```bash
git pull
dvc pull
```

Always pull both. `git pull` grabs the updated pointer file (`data.dvc`), and `dvc pull` downloads the actual data it points to.

### Troubleshooting

- **"Permission denied" on dvc pull** -- You need access to the shared Google Drive folder. Ask JR.
- **Authentication issues** -- Run `dvc remote modify gdrive gdrive_acknowledge_abuse true` and try again. If it still fails, try clearing cached credentials: `rm -rf ~/.cache/pydrive2/`
- **Slow download** -- The dataset is ~10GB. On NYU wifi it should take a few minutes. On a home connection it may take longer.
- **Disk space** -- DVC keeps a local cache in `.dvc/cache/`. The data effectively takes ~20GB (cache + working copy). You can run `dvc gc -w` to clean up the cache after pulling, which brings it back down to ~10GB.
