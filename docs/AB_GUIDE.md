# Technique A vs Technique B runner: guide (Sameer)

What this adds (the signed-off M1 to M4 files are NOT changed):
- `src/run_ab.py`: runs Lookup, Decision Tree, Logistic Regression and Random Forest under Technique A (stratified random
  80/20) and Technique B (chronological block split inside each recording) and writes the team's
  `results_table_template.csv` format exactly (200 rows, same header, same order).
- `src/splits_ab.py`: both splits, built from the rules in the Master Context Pack (6.6, 6.7, 6.9).
- `src/data_ab.py`: reads the repo's `frames.parquet` and removes the duplicated attack-free tails (6.5).
- `src/lookup.py`: the exact-signature lookup baseline (predicts Normal for a signature not seen in training, D20).
- `src/summarise_ab.py`: mean and standard deviation over seeds, from the filled table.
- `src/synthetic_ab.py`, `config_ab.yaml`, `tests/test_splits_ab.py`, `tests/test_run_ab.py`.

## Step 1: put the files in the project (Colab)
Upload `ab_protocols_update.zip` into the project folder on Drive, then:
    %cd /content/drive/MyDrive/FYDP/files/fydp_models_sameer/fydp_models
    !unzip -o ab_protocols_update.zip
    !pip install -q -r requirements.txt
It only adds files. `config.yaml` is not touched, so your frozen settings stay as they are.

## Step 2: tests (about 1 minute)
    !python -m pytest -q
Expect 33 passed (17 old + 16 new). Note: `test_real_run_refused_until_frozen_and_tolerance_fixed` (old runner test) will
FAIL as soon as you set both flags in `config.yaml` to true, because it expects the refusal. That is expected; change
the flags only for real runs, or tell me and I will make that test independent of the config.

## Step 3: dry run on synthetic recordings
    !python -m src.run_ab --synthetic --models Lookup DT --seeds 42 43 --out-dir results/ab_dry_run
Check: `[ok]` lines, then "all self-checks passed". Delete `results/ab_dry_run` afterwards. The numbers mean nothing.

## Step 4: get the real frame table
You need the repo pipeline's table `frames.parquet` (columns ts, id, dlc, d0..d7, label, source, flag, sig_id). It is not in Git.
Ask Hafiz for it on Drive, or build it from the five raw files with `python src/build_dataset.py`
(set the environment variable `FYDP_DATA` to the folder that holds `raw/`). 17,558,346 rows is expected.

## Step 5: check my splits against the team's numbers (trains nothing, so it is safe now)
    !python -m src.run_ab --table /content/drive/MyDrive/<path>/frames.parquet --splits-only --known-checks --out-dir results
Every line should say PASS: Technique B test frames per class (Normal 2,344,847; DoS 108,623; Fuzzy 76,121; RPM 135,084;
Gear 118,490), the cut times (DoS 1864.503 s, Fuzzy 1972.794 s, RPM 1566.145 s, Gear 1562.469 s, ambient 405.149 s),
exact A test counts, and the seen shares (Normal 0.971 under A, 0.9145 under B). It also prints how many frames each
recording lost in sanitising: 891,068 for each of dos, fuzzy, gear, rpm and 0 for normal. If something says DIFF, stop and
send me the output: either my code or the team's number is wrong, and that must be settled before any model runs.

## Step 6: the real run (only after the two flags are true)
The runner refuses real runs until `frozen: true` and `validity.tolerance_fixed: true` in `config.yaml`
(the supervisor must confirm the validity tolerance first). Then:
    !python -m src.run_ab --table /content/drive/MyDrive/<path>/frames.parquet --template results/results_table_template.csv --models Lookup DT LR --techniques A B
Start with the cheap models, Random Forest last (`--models RF`). Finished runs are skipped if Colab disconnects: start the same command again.
Put the team's `results_table_template.csv` in `results/` first (or pass `--template`). Output: `results/results_table_filled_<time>.csv`.
Then: `!python -m src.summarise_ab results/results_table_filled_<time>.csv`

## How the template is filled
- Row key = (technique, seed, model, label). tp, fp, fn are counted over the whole test side. A test frame is "seen" if its exact
  signature occurs in the training side; tp_seen + tp_unseen = tp and fn_seen + fn_unseen = fn.
- Technique B has one fixed split. Lookup and Logistic Regression (lbfgs) have no randomness, so they are run once and the rows are
  copied to the other four seeds (the log says so). Decision Tree and Random Forest run once per seed.
- Self-checks before the file is written: every model sees the same number of test frames per class, sum of fp = sum of fn,
  seen + unseen = total. Any failure is printed.
- Model names in the table: Lookup, DecisionTree, LogisticRegression, RandomForest.

## Things to agree with the team (not decided by me)
1. Hyperparameters: Hafiz's pilot placeholders (RF 50 trees, balanced_subsample) differ from the frozen set (RF 100 trees,
   max_samples 0.5, balanced). One set must be written in DECISIONS.md before the real run.
2. The supervisor's open questions (Master Context Pack 2.4): tail removal, validity tolerance, embargo, Logistic Regression
   encoding of the identifier, variance of Technique B.
3. The training side is about 11.2 million frames (not 14 million), so the timing estimates are on the safe side.
