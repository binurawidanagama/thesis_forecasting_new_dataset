# Efficient Hybrid Learning-and-Reasoning for Multi-Horizon Multivariate Time-Series Forecasting under Resource Constraints
### dCeNN-ELM-ASP vs. LSTM/CNN on Austrian 15-Minute Weather and Energy Data

> **Key idea:** this repository contains a neuro-symbolic forecasting framework for **15-minute resolution Austrian weather and energy data**. In the **current exported results**, the **dCeNN → ELM → ASP path is trained and benchmarked for the weather task**, while the **energy benchmark results currently included are baseline-only (CNN/LSTM)**. The repo still contains the energy configuration and pipeline scaffolding for the proposed approach.

---

## Table of Contents
1. [Overview](#overview)
2. [What this repository predicts](#what-this-repository-predicts)
3. [Architecture overview](#architecture-overview)
4. [Repository structure](#repository-structure)
5. [Dataset and files](#dataset-and-files)
6. [Feature engineering](#feature-engineering)
7. [Train / validation / test split](#train--validation--test-split)
8. [Models and symbolic layer](#models-and-symbolic-layer)
9. [How to run the code](#how-to-run-the-code)
10. [Benchmarking and visualisation](#benchmarking-and-visualisation)
11. [Selected benchmark results](#selected-benchmark-results)
12. [Outputs produced by the pipeline](#outputs-produced-by-the-pipeline)
13. [Notes on naming conventions](#notes-on-naming-conventions)
14. [Dependencies](#dependencies)
15. [Thesis context](#thesis-context)
16. [Suggested citation / description](#suggested-citation--description)

---

## Overview

This repository contains the full experimental pipeline for a thesis on **Efficient Hybrid Learning-and-Reasoning for Multi-Horizon Multivariate Time-Series Forecasting under Resource Constraints: A Comparative Evaluation of dCeNN–ELM–ASP approach with LSTM and CNN on Weather and Energy Data**. The focus is not just raw prediction accuracy, but the trade-off between:

- predictive performance,
- computational cost,
- inference latency, and
- rule-based plausibility correction.

The codebase includes:

- data preparation and feature engineering,
- the proposed dCeNN + ELM + ASP workflow,
- CNN and LSTM baselines,
- ASP-based post-processing logic for energy and weather tasks,
- benchmark aggregation,
- ranking, Pareto analysis, and plotting utilities.

> **Current status note:**weather** includes dCeNN+ELM results, while **energy** currently includes **CNN/LSTM baseline results only**.

---

## What this repository predicts

### Energy task
The current **energy configuration** predicts a **single target**:

- `load_mw`

The energy pipeline uses calendar features plus four weather drivers:

- `temperature_2m_C`
- `precipitation_mm`
- `mean_global_radiation`
- `mean_wind_speed`

### Weather task
The current **weather configuration** predicts four targets:

- `temperature_2m_C`
- `precipitation_mm`
- `mean_global_radiation`
- `mean_wind_speed`

So this repository is **not** the old total-energy setup from the reference repo. It has been adapted to the new dataset and task definitions used in this repo.

> **Important implementation note:** the repo contains an **energy configuration and scripts for the proposed pipeline**, but the **repo does not contain trained energy dCeNN+ELM results**. In that export, energy comparisons are between **CNN** and **LSTM** baselines. Energy configurations are yet to be trained.

---

## Architecture overview

```text
Raw CSV data + Austrian holiday calendar
                │
                ▼
┌──────────────────────────────────────────────┐
│ 01 · Data preparation / feature engineering  │
│    scripts/prepare_data.py                   │
│    src/dataio/*                              │
└───────────────────────┬──────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────┐
│ 02 · dCeNN encoder training                  │
│    scripts/run_energy_full.py                │
│    scripts/run_weather_full.py               │
│    src/models/dcenn.py                       │
└───────────────────────┬──────────────────────┘
                        │ latent representation
                        ▼
┌──────────────────────────────────────────────┐
│ 03 · ELM forecasting head                    │
│    ELM weights / deployment payload saved    │
│    as .npz / .pt artifacts                   │
└───────────────────────┬──────────────────────┘
                        │ raw predictions
                        ▼
┌──────────────────────────────────────────────┐
│ 04 · ASP symbolic repair / plausibility      │
│    scripts/run_energy_asp.py                 │
│    scripts/run_weather_asp.py                │
│    src/asp/*.lp                              │
└───────────────────────┬──────────────────────┘
                        │ repaired predictions
                        ▼
┌──────────────────────────────────────────────┐
│ 05 · Evaluation / benchmarking / plots       │
│    scripts/run_eval.py                       │
│    scripts/collect_benchmarks.py             │
│    scripts/make_benchmark_table.py           │
│    scripts/plot_pareto.py                    │
│    scripts/rank_and_plot_benchmarks.py       │
└──────────────────────────────────────────────┘
```

---

## Repository structure

```text
thesis_forecasting_new_dataset/
├── configs/
│   ├── default.yaml
│   ├── energy_full.yaml
│   └── weather_full.yaml
│
├── data/
│   ├── raw/
│   │   ├── gen_dataset.csv
│   │   └── weather_data_15min.csv
│   ├── processed/
│   ├── interim/
│   ├── interim_energy_full/
│   ├── interim_weather_full/
│   └── AT_public_holidays_2020_2025.csv
│
├── src/
│   ├── asp/
│   │   ├── batch_facts.lp
│   │   ├── core_asp.lp
│   │   ├── energy_physics.lp
│   │   ├── weather_physics.lp
│   │   └── related ASP programs
│   ├── dataio/
│   ├── eval/
│   ├── inference/
│   ├── models/
│   ├── train/
│   └── utils/
│
├── scripts/
│   ├── prepare_data.py
│   ├── run_energy_full.py
│   ├── run_weather_full.py
│   ├── run_energy_asp.py
│   ├── run_weather_asp.py
│   ├── run_cnn_baseline.py
│   ├── run_lstm_baseline.py
│   ├── run_eval.py
│   ├── collect_benchmarks.py
│   ├── make_benchmark_table.py
│   ├── plot_pareto.py
│   ├── rank_and_plot_benchmarks.py
│   ├── cnn_lstm_visualize.py
│   └── dcenn_visualize.py
│
├── outputs/
│   ├── predictions/
│   └── eval/
│
├── outputs_weather_full/
├── artifacts_cnn_baseline/
├── artifacts_lstm_baseline/
├── thesis_plots_final_15min/
│   └── COMPARE_ALL/
│
├── benchmarks_master.csv
├── benchmark_table_rmse_ratio.csv
├── benchmark_table_latency_ms.csv
├── benchmark_winners_rmse_ratio.csv
├── benchmark_pareto_all.png
├── pareto_points.png
├── rank_compute_latency.png
├── rank_compute_only.png
├── rank_latency.png
├── rank_overall.png
└── model_ranking_with_compute_latency.csv
```

---

## Dataset and files

The repository expects the main raw inputs in:

- `data/raw/gen_dataset.csv`
- `data/raw/weather_data_15min.csv`
- `data/AT_public_holidays_2020_2025.csv`

The current configs indicate a **15-minute time resolution** and use `Time (UTC)` as the timestamp column.

### Current task definitions from config

**Energy task**
- Input features include cyclical time features, holiday indicators, weather variables, and `load_mw`
- Target features: `load_mw`

**Weather task**
- Target features:
  - `temperature_2m_C`
  - `precipitation_mm`
  - `mean_global_radiation`
  - `mean_wind_speed`

---


## Feature engineering

The feature design is intentionally **calendar-aware** and **periodicity-aware**. Instead of feeding raw timestamps directly into the models, the pipeline uses engineered variables that make recurring temporal structure easier to learn and avoid artificial discontinuities.

### Why not use raw timestamps?
A raw hour value such as `23` followed by `0` creates a fake numerical jump even though those times are adjacent. Cyclical encodings solve that problem by mapping periodic variables onto a circle, so midnight is close to 23:45 rather than far away.

### Core engineered temporal and calendar features

| Feature | Type | Description |
|---|---|---|
| `hour_sin`, `hour_cos` | Cyclical | Sine/cosine encoding of time-of-day; avoids the 23:00 → 00:00 discontinuity. |
| `dow_sin`, `dow_cos` | Cyclical | Sine/cosine encoding of day-of-week patterns. |
| `month_sin`, `month_cos` | Cyclical | Sine/cosine encoding of seasonal monthly structure. |
| `is_weekend` | Boolean | Flag for Saturdays and Sundays. |
| `is_public_holiday` | Boolean | Flag for official Austrian public holidays. |
| `is_special_day` | Boolean | Flag for bridging days or non-statutory special observance days used in the preprocessing logic. |

### Task-specific inputs

For the **energy task**, the current configuration combines:
- engineered temporal features,
- holiday / weekend indicators,
- weather drivers (`temperature_2m_C`, `precipitation_mm`, `mean_global_radiation`, `mean_wind_speed`), and
- the target-linked energy series `load_mw`.

For the **weather task**, the configuration keeps the same time-aware structure and predicts four physical weather variables jointly:
- `temperature_2m_C`
- `precipitation_mm`
- `mean_global_radiation`
- `mean_wind_speed`

This design makes the repository suitable for **multi-horizon forecasting under periodic seasonal structure**, while keeping the features interpretable enough for thesis reporting and ablation discussion.

---
## Train / validation / test split

Both `energy_full.yaml` and `weather_full.yaml` use the following split boundaries:

- **Train until:** `2024-06-30 23:45:00`
- **Validation until:** `2024-09-30 23:45:00`
- **Test until:** `2024-12-31 23:45:00`

This keeps the evaluation temporally clean and consistent across models.

---

## Models and symbolic layer

### 1) dCeNN + ELM
The main proposed method is a hybrid pipeline where:

- **dCeNN** acts as the representation learner / encoder,
- **ELM** provides fast forecasting on top of the learned latent space,
- deployment artifacts are saved separately from training artifacts,
- parameter counts, model size, CPU time, latency, and peak RAM are tracked for thesis-grade benchmarking.

### 2) Baselines
Two neural baselines are included for fair comparison:

- **CNN / TCN baseline**
- **LSTM baseline**

These baseline runs are stored under:

- `artifacts_cnn_baseline/`
- `artifacts_lstm_baseline/`

### 3) ASP repair layer
The symbolic post-processing stage applies logic rules from:

- `src/asp/energy_physics.lp`
- `src/asp/weather_physics.lp`
- related ASP support files in `src/asp/`

This layer is used to repair or flag raw predictions that violate domain constraints.

---

## How to run the code

### 1. Clone and install
```bash
git clone https://github.com/binurawidanagama/thesis_forecasting_new_dataset.git
cd thesis_forecasting_new_dataset
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
export PYTHONPATH=$PWD      # Linux / macOS
# set PYTHONPATH=%CD%       # Windows CMD
```

### 2. Make sure the raw files are in place
Place the dataset files in:

```text
data/raw/gen_dataset.csv
data/raw/weather_data_15min.csv
data/AT_public_holidays_2020_2025.csv
```

### 3. Run the proposed dCeNN + ELM pipeline
Example energy runs:

```bash
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 96 --horizon 12
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 96 --horizon 24
python scripts/run_energy_full.py --config configs/energy_full.yaml --lookback 96 --horizon 72
```

Example weather runs:

```bash
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 96 --horizon 12
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 96 --horizon 24
python scripts/run_weather_full.py --config configs/weather_full.yaml --lookback 96 --horizon 72
```

You can repeat the same pattern for the other thesis settings, typically using:

- lookbacks: `96`, `288`, `672`
- horizons: `12`, `24`, `72`

### 4. Apply ASP post-processing
```bash
python scripts/run_energy_asp.py --config configs/energy_full.yaml --lookback 96 --horizon 12
python scripts/run_weather_asp.py --config configs/weather_full.yaml --lookback 96 --horizon 12
```

### 5. Run baseline models
```bash
python scripts/run_cnn_baseline.py
python scripts/run_lstm_baseline.py
```

### 6. Aggregate benchmark results
```bash
python scripts/collect_benchmarks.py
python scripts/make_benchmark_table.py
python scripts/plot_pareto.py
python scripts/rank_and_plot_benchmarks.py
```

### 7. Visualise forecast trajectories
```bash
python scripts/cnn_lstm_visualize.py
python scripts/dcenn_visualize.py
```

---

## Benchmarking and visualisation

The repository already includes consolidated benchmark outputs such as:

- `benchmarks_master.csv`
- `benchmark_table_rmse_ratio.csv`
- `benchmark_table_latency_ms.csv`
- `benchmark_winners_rmse_ratio.csv`
- `model_ranking_with_compute_latency.csv`
- `benchmark_pareto_all.png`
- `pareto_points.png`
- `rank_compute_latency.png`
- `rank_compute_only.png`
- `rank_latency.png`
- `rank_overall.png`

The benchmark scripts compare models on:

- **accuracy** (MAE, RMSE, sMAPE, RMSE ratio),
- **training cost**,
- **inference cost**,
- **latency per sample**,
- **peak RAM**, and
- **deployable parameter count / artifact size**.

The plotting utilities generate global ranking and Pareto-style views of the accuracy–efficiency trade-off.

---


## Selected benchmark results

The repository already contains consolidated benchmark files such as `benchmarks_master.csv`, `benchmark_table_rmse_ratio.csv`, `benchmark_table_latency_ms.csv`, and the associated ranking / Pareto plots. The table below highlights the **best-RMSE configuration found for each task and forecast horizon** from the uploaded benchmark CSV.

### Best accuracy by task and horizon

> **Reading this table carefully matters:** for **ENERGY**, the current benchmark export contains **baseline-only results** (CNN/LSTM). For **WEATHER**, it contains **CNN, LSTM, and raw dCeNN+ELM** runs.

| Task | Horizon | Best model | Lookback | RMSE | MAE | Latency (ms/sample) | Deploy params |
|---|---:|---|---:|---:|---:|---:|---:|
| ENERGY | 12 | CNN | 96 | 4.919 | 3.915 | 0.965 | 52,652 |
| ENERGY | 24 | CNN | 288 | 5.249 | 4.128 | 2.327 | 53,432 |
| ENERGY | 72 | LSTM | 96 | 5.859 | 4.533 | 0.570 | 131,073 |
| WEATHER | 12 | CNN | 288 | 23.679 | 6.527 | 1.498 | 54,800 |
| WEATHER | 24 | CNN | 288 | 29.341 | 8.376 | 1.516 | 57,920 |
| WEATHER | 72 | CNN | 288 | 32.525 | 10.744 | 1.628 | 70,400 |

### What these results mean

- **Energy forecasting is very strong relative to the benchmark baseline.** The best energy runs achieve RMSE ratios of approximately **0.213**, **0.154**, and **0.140** for 12h, 24h, and 72h respectively, which corresponds to roughly **78.7%**, **84.6%**, and **86.0%** lower RMSE than the reference baseline used in the benchmark table.
- **CNN is the most consistent top performer in the current benchmark export.** It gives the lowest RMSE for all three weather horizons and for energy at 12h and 24h. The **LSTM** slightly edges out the others on **energy, 72h horizon**.
- **Do not over-interpret the energy section as a full proposed-method comparison.** In this CSV, the energy rows are a **baseline-vs-baseline comparison only**, because trained energy dCeNN+ELM results are not present yet.
- **Inference speed tells a different story from accuracy.** For the energy task, the fastest runs are the **LSTM 96-lookback** models with latency around **0.51–0.57 ms/sample**. For the weather task, the fastest runs in this export are the **raw dCeNN+ELM** models at **~0.52 ms/sample**, but these do **not** provide the best RMSE.
- **The current raw weather dCeNN+ELM configuration is not the accuracy leader in this CSV.** At lookback 96 it remains reasonably fast but yields RMSE values of **38.660**, **42.452**, and **39.700** for 12h, 24h, and 72h, while the lookback-288 raw setting shows clear instability at 12h and 24h.

### Benchmark artefacts already included in the repository

- `benchmark_pareto_all.png` and `pareto_points.png` for accuracy–efficiency trade-offs
- `rank_overall.png`, `rank_latency.png`, and `rank_compute_latency.png` for model ranking
- `benchmark_winners_rmse_ratio.csv` for winner tracking by metric
- `model_ranking_with_compute_latency.csv` for multi-criteria model scoring

---
## Outputs produced by the pipeline

### General outputs
The generic output folder contains:

- `outputs/predictions/test_predictions.parquet`
- `outputs/predictions/test_predictions_asp.parquet`
- `outputs/eval/metrics_test.json`

### Weather dCeNN outputs
Each weather experiment folder such as `outputs_weather_full/LB96_H12/` contains artifacts including:

- `base_metrics.json`
- `dcenn_weather_deploy.pt`
- `dcenn_weather_train.pt`
- `elm_betas.npz`
- `params_accounting.json`
- `raw_weather.parquet`
- `truth_weather.parquet`

### Baseline outputs
Baseline experiment artifacts are organised by lookback / horizon / task under:

- `artifacts_cnn_baseline/`
- `artifacts_lstm_baseline/`

### Comparison plots
Combined forecast comparison figures are stored under:

- `thesis_plots_final_15min/COMPARE_ALL/`

---

## Notes on naming conventions

This repository currently contains **two naming styles** for horizons / artifacts:

1. **Canonical thesis benchmark labels** such as `12`, `24`, `72`
2. **Legacy 15-minute-step style folders** such as `H48`, `H96`, `H288`

That means you may see older artifact folders like:

- `artifacts_cnn_baseline_48_96_288H/`

alongside the current consolidated benchmark tables that use the cleaner thesis labels.

In plain English: the repo has a little naming archaeology in it. The pipeline is still usable, but the README should acknowledge the fossils instead of pretending they are decorative pottery.

---

## Dependencies

The current `requirements.txt` lists:

- `numpy`
- `pandas`
- `pyyaml`
- `scikit-learn`
- `tqdm`
- `python-dateutil`
- `pytz`
- `joblib`
- `pyarrow`
- `holidays`
- `clingo`

`clingo` is required for the ASP reasoning layer.

---

## Thesis context

This repository is intended for the comparative evaluation of:

- **dCeNN + ELM** (raw)
- **dCeNN + ELM + ASP** (repaired / constrained)
- **CNN baseline**
- **LSTM baseline**

across multi-horizon forecasting settings on Austrian weather and energy data under resource constraints.

The emphasis is on a **reproducible end-to-end research workflow**: from raw CSV ingestion, to forecasting, to symbolic repair, to benchmark aggregation and thesis-ready plots / tables.

---

## Suggested citation / description
