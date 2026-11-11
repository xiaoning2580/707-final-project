# Shared Project Environment

## One installation entry point

Use Python **3.11.18** and the root [requirements.txt](../requirements.txt)
for the data preparation, EDA, LASSO, and random-forest scripts. This is the
only project dependency list. It pins six main analysis packages and twelve
supporting packages from the validated modeling environment.

| Main package | Version |
| --- | --- |
| numpy | 2.4.6 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| scipy | 1.17.1 |
| matplotlib | 3.11.2 |
| joblib | 1.6.0 |

## Create and activate the environment

Run these commands from the repository root on macOS/Linux:

```bash
conda create --prefix "$PWD/.conda-707-aligned" python=3.11.18 pip=26.2.2 -y
conda activate "$PWD/.conda-707-aligned"
python -m pip install -r requirements.txt
python -m pip check
```

If the environment already exists, activate it and use the install/check
commands. The environment directory is local and ignored by Git. Keep it
in place; moving an installed Conda environment can break executable paths.

The commands in the analysis guides assume this environment is activated.
On this Mac, `.conda-707-aligned/bin/python` can also be used directly.
For example:

```bash
python scripts/06_build_features.py
python scripts/07_training_eda.py
python scripts/11_train_regularized_models.py --tune-lasso
python scripts/11_train_regularized_models.py --predict-test
python scripts/12_train_random_forests.py --mode all --jobs 4
```

Place the agreed `split_assignments.csv` in `data/processed/` before building
features. See [data usage](data_usage.md) for the inputs and handoff rules.

## Provenance and verification

- The main versions match Yixuan's supplied 2026-11-10 environment snapshot
  and Kaiyan's updated modeling specification in commit `fc6a813`.
- Yixuan's original snapshot and data notes are retained locally under
  `data/handoff/yixuan_2026-11-10/`; the snapshot describes export-time versions.
- The received data usage and preparation notes matched the repository copies.
- Mai regenerated features and refitted all three random forests with Python
  3.11.18 and these versions. The shared split was preserved; regenerated
  inputs matched the prior inputs by ID and value.
- The latest random-forest outputs record their versions and input hashes in
  `results/tables/random_forests/rf_selection.json`. Model and prediction
  readback checks passed. Only the latest outputs are retained.
- The 18 package pins match the installed project environment. This specifies
  the Python runtime dependencies, not identical operating systems or all
  packages installed on each teammate's computer. Mai's run used macOS 26.5,
  arm64. A teammate's exact prepared feature files have not been compared
  because those files were not included in the supplied handoff documents.

Future dependency changes should update the root requirements file and this
guide together, followed by checks of the affected analysis scripts.
