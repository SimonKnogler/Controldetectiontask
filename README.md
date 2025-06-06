This repository contains experiments, data processing pipelines and GAN models
for generating and presenting short **motion snippets**. These snippets are used
in a psychophysics experiment referred to as **Control Detection Task (CDT)**.

## Repository layout

- `Experiment/` and `Main Experiment/` – PsychoPy implementations of the CDT.
  The `CDT.py` script in these folders presents motion snippets and collects
  participants' responses.
- `Motion Library/` – processed snippet library (`core_pool.npy`) together with
  clustering metadata (`core_pool_feats.npy`, `core_pool_labels.npy`,
  `scaler_params.json` and `cluster_centroids.json`). This folder also contains
  `motionlib_create_filtered.py` which builds the library from raw donor
  recordings.
- `Scripts/` – helper utilities for recording raw donor motions and assembling
  the master library.
- `Technical Reports/` – PDF and figures describing the motion library.
- `Trial Demo/` – short videos demonstrating the task.

## The Control Detection Task

`CDT.py` presents a moving dot controlled either fully or partially by the
participant. After an initial demo phase the participant's movement features are
compared with precomputed cluster centroids from the motion library. Subsequent
trials draw snippets from the cluster that best matches the participant's style,
ensuring motion cues are tailored to them. Responses are saved as CSV files in
the experiment's `data/` directory.

## Motion snippet creation

Raw motion was recorded using the PsychoPy script in `Motion Library/Experiment Pavlovia`. Individual
recordings were combined, filtered and clustered by `Motion Library/motionlib_create_filtered.py`:
snippets are sliced into 3‑second segments, low‑movement samples are discarded
and K‑means clustering groups them into six style clusters. Scaler parameters and
cluster centroids are saved for style‑matching during the CDT.

## GAN Scripts

- `gan_motion_generator.py` – trains a simple unconditional GAN on `core_pool.npy`.
- `generate_gan_snippets.py` – loads a trained GAN and visualises generated snippets.
- `conditional_gan_motion_generator.py` – trains a conditional GAN using snippet
  labels from `core_pool_labels.npy`. The generator and discriminator both receive
  cluster labels so new snippets can be generated conditioned on a specific
  style cluster.
- `generate_cgan_snippets.py` – loads a conditional GAN generator and
  produces snippets for a chosen style label.

Example usage:

```bash
python generate_cgan_snippets.py --label 2 --num-samples 5
```

To run tests:

```bash
pytest -q
```

## Simulation mode

Both `Experiment/CDT.py` and `Main Experiment/CDT.py` now provide an optional
*Simulate* checkbox in the startup dialog. When enabled, the task generates
artificial mouse movements and key presses so that CSV data can be produced
without a human participant. This is useful for quickly testing analysis
pipelines during piloting.

## Data analysis

An example R script for analysing output CSV files is available in `Scripts/analysis.R`.
It aggregates participant data, computes accuracy, d-prime and metacognitive measures using
Fleming's HMeta-d toolbox, performs the repeated measures ANOVA and mixed-effects
models described in the main analysis plan, and saves several summary figures under `Plots/`.

Run the script from the repository root with:

```bash
Rscript Scripts/analysis.R
```
