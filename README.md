# Copula-Comparisons

Copula-Comparisons is a unified benchmarking framework for comparing synthetic tabular data generators. It evaluates four copula generators (BetaCopula, CheckerboardCopula, GaussCopula and VineCopula) against established deep generative baselines (TabDDPM, CTAB-GAN, CTAB-GAN+ and TVAE) using a single, consistent evaluation pipeline so that all numbers are directly comparable. It contains the code and stored results for the paper *The Empirical Beta Copula as a Tabular Data Synthesizer: Utility, Inference Fidelity, and Cost Against Deep Generative Models* (Eckert and Molinari).

## Setup

### 1. Create the Conda environment

NOTE TO THE PEOPLE WORKING ON THIS:

The project requires **Python 3.9.7** and Conda. Much of the original config from the TabDDPM GitHub has been changed to run locally on CPU. If you want to use CUDA, refer to the upstream TabDDPM repository and swap in your own configs — **do not push CUDA-specific changes**, as it will break development on machines without a GPU.

```bash
conda create -n copula-comparisons python=3.9.7
conda activate copula-comparisons
```

### 2. Install dependencies

```bash
conda config --append channels conda-forge
conda install numpy catboost scikit-learn xgboost tomli-w tomli \
    dython icecream optuna pyarrow skorch imbalanced-learn \
    rdt invoke ctgan pytorch category_encoders matplotlib
pip install libzero rtdl pyvinecopulib==0.7.6
```

Then set the required environment variables (update `REPO_DIR` to the absolute path of this repository):

```bash
conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}

# Reactivate so the variables take effect
conda deactivate
conda activate copula-comparisons
```

If the `conda env config vars set` commands fail, update Conda first (`conda update conda`) and retry.

### 3. Download the datasets

The datasets are too large for GitHub, so they are hosted externally. **Do not commit the `data/` directory.**

```bash
conda activate copula-comparisons
cd $PROJECT_DIR
wget "https://www.dropbox.com/s/rpckvcs3vx7j605/data.tar?dl=0" -O data.tar
tar -xvf data.tar
```

This creates a `data/` directory with the following 15 datasets:

| Dataset | Abbreviation | Task |
|---|---|---|
| abalone | AB | regression |
| adult | AD | binclass |
| buddy | BU | multiclass |
| california | CA | regression |
| cardio | CAR | binclass |
| churn2 | CH | binclass |
| diabetes | DI | binclass |
| fb-comments | FB | regression |
| gesture | GE | multiclass |
| higgs-small | HI | binclass |
| house | HO | regression |
| insurance | IN | regression |
| king | KI | regression |
| miniboone | MI | binclass |
| wilt | WI | binclass |

The TabDDPM benchmark also lists a `default` dataset, which is not in this archive and is not used.

## Running the Benchmark

The main entry point is `benchmark.py`. It fits or loads every method and evaluates all of them through the same pipeline. The four copulas are fitted on the fly. TabDDPM, CTAB-GAN, CTAB-GAN+ and TVAE are sampled from the models trained by `train_all_baselines.py`, which uses the per-dataset tuned configuration in `exp/{dataset}/{method}/config.toml` distributed with the TabDDPM repository.

Training for the deep baseline models is done using `train_all_baselines.py`. Each trained model is written to `exp/{dataset}/{method}/`, and its training time and on-disk size are recorded in `results/training_cost/train_metrics.csv`. Trained models are not tracked by git; the training measurements used in the paper are stored in that CSV. Training runs sequentially on purpose, so that the recorded times are not distorted by other jobs competing for the CPU. Models that are already trained are skipped unless `--force` is passed.

```bash
python train_all_baselines.py
python train_all_baselines.py --datasets diabetes wilt --methods tabddpm tvae
```

The paper's results are stored in `results/benchmarks/benchmark_results.json`. Because the benchmark resumes from that file, a plain run finds every (dataset, method) pair already done. Pass `--no_resume`, or point `--output_dir` at a new folder, to recompute.

### Basic usage

Run every method on every dataset with default settings (5 generation seeds x 10 evaluation seeds = 50 runs per method):

```bash
python benchmark.py --output_dir results/rerun/
```

### Specifying datasets

Run only on specific datasets by passing their names with `--datasets`:

```bash
python benchmark.py --datasets diabetes adult wilt
```

### Disabling specific methods

You can skip individual copulas or all baselines:

```bash
# Skip all baselines, only run the copulas
python benchmark.py --no_baselines

# Skip the vine copula (the slowest copula to fit)
python benchmark.py --no_vine_copula

# Run only the beta and checkerboard copulas (and the Real reference)
python benchmark.py --no_baselines --no_gaussian_copula --no_vine_copula

# Show what would run (cached vs. to-do) without computing anything
python benchmark.py --output_dir results/benchmarks/ --dry_run
```

Available flags:
| Flag | Effect |
|---|---|
| `--no_baselines` | Skip all baseline methods (TabDDPM, TVAE, CTAB-GAN, CTAB-GAN+) |
| `--no_beta_copula` | Skip BetaCopula |
| `--no_gaussian_copula` | Skip GaussCopula |
| `--no_checkerboard_copula` | Skip CheckerboardCopula |
| `--no_vine_copula` | Skip VineCopula (needs `pyvinecopulib`) |

### Running specific baselines

Instead of running all baselines, you can choose which ones to include:

```bash
python benchmark.py --baselines tabddpm tvae
```

Valid baseline names: `tabddpm`, `tvae`, `ctabgan`, `ctabgan_plus`

CTAB-GAN requires a classification target, so it has configurations only for the nine classification datasets.

### Controlling seeds

Adjust the number of generation and evaluation seeds (default: 5 and 10 respectively):

```bash
python benchmark.py --n_gen_seeds 3 --n_eval_seeds 5
```

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/` | Path to the dataset directory |
| `--exp_dir` | `exp/` | Path to experiment directory with baseline configs and trained models |
| `--output_dir` | `results/benchmarks/` | Where to save the JSON results file and the per-metric CSVs |
| `--tuned_models_dir` | `tuned_models/` | Directory for tuned CatBoost model parameters |
| `--datasets` | all 15 | Space-separated list of dataset names to evaluate |
| `--n_gen_seeds` | 5 | Number of generation seeds |
| `--n_eval_seeds` | 10 | Number of evaluation seeds |
| `--baselines` | all | Space-separated list of specific baselines to run |
| `--n_workers` | `cpu_count - 1` | Parallel (dataset, method) tasks; use 1 for sequential |
| `--no_resume` | off | Recompute pairs already present in `benchmark_results.json` |
| `--dry_run` | off | Print the cached / to-run plan and exit without writing |

## Evaluation Protocol

The benchmark uses two utility protocols, plus statistical fidelity, faithfulness, privacy and inference-fidelity metrics:

- **Protocol 1 (P1):** Average ML efficiency over weak models (Decision Tree, Random Forest, Logistic Regression/Ridge, MLP)
- **Protocol 2 (P2):** ML efficiency with a tuned CatBoost model (Macro F1 for classification, R2 for regression)
- **Statistical fidelity:** correlation-matrix L2 distance (CorrL2), mixed-type association-matrix L2 distance (AssocL2), standardized Wasserstein distance (WD), Jensen-Shannon divergence (JS), and the pMSE ratio of a CART real-vs-synthetic classifier against real test rows (ideal value 1)
- **Faithfulness:** alpha-precision, beta-recall and authenticity with k-nearest-neighbor supports (k = 5)
- **Privacy:** Distance to Closest Record (DCR), reported as the ratio of the mean (DCR) and of the fifth percentile (DCR5%) to the same quantities for real test rows — higher means more private
- **Inference fidelity:** for an analyst GLM of the target on all features, the 95% confidence-interval overlap, the coverage of the real estimate by the synthetic interval, and sign-and-significance agreement

## Output

Results are saved to `results/benchmarks/benchmark_results.json` (updated after each (dataset, method) task completes). When the benchmark finishes, a formatted summary table is printed to the console showing all methods side-by-side. Per-metric CSVs (`p1_table.csv`, `p2_table.csv`, `corr_l2_table.csv`, `assoc_l2_table.csv`, `wd_table.csv`, `js_table.csv`, `pmse_ratio_table.csv`, `alpha_precision_table.csv`, `beta_recall_table.csv`, `authenticity_table.csv`, `dcr_table.csv`, `dcr_ratio_table.csv`, `dcr_p5_ratio_table.csv`, `ci_overlap_table.csv`, `coverage_table.csv`, `sig_agree_table.csv`) and an aggregated cross-dataset summary (`aggregate_table.csv`) are written to `--output_dir` (default `results/benchmarks/`).

## Models compared

The benchmark compares four copula generators. All four use the empirical distribution of each column as its marginal and differ in the dependence model:

| Generator | Dependence model | Categorical columns |
|---|---|---|
| BetaCopula | empirical beta copula: draw a training row, replace each rank r by a Beta(r, n+1-r) draw | one-hot, decoded from latent scores |
| CheckerboardCopula | checkerboard copula: draw a training row, replace each rank r by a uniform draw in the cell [(r-1)/n, r/n] | one-hot, decoded from latent scores |
| GaussCopula | Gaussian copula fitted to the normal scores | one-hot, decoded from latent scores |
| VineCopula | R-vine from `pyvinecopulib` (one-parameter families, Kendall's tau inversion, mBICV selection, automatic truncation) | integer codes with discrete margins |

BetaCopula and CheckerboardCopula differ only in the kernel applied to the ranks, so the pair isolates what the beta smoothing contributes.

The baselines are:

- **TabDDPM, CTAB-GAN, CTAB-GAN+, TVAE** — deep generative baselines, trained via `train_all_baselines.py` with the per-dataset tuned configurations. CTAB-GAN is evaluated on the classification datasets only.

## Additional tooling

- **`aggregate_table.csv`** — a cross-dataset rollup written alongside the per-metric tables. Headline column is the correlation-matrix L2 distance (mean ± std over datasets), with P1 and P2 ML-efficiency shown beside it.

- **Timing / space / parameter profiling** — `train_all_baselines.py` measures, per (dataset, model), the training time and the on-disk size of the trained model (for the copulas, the size of the pickled fitted object), and writes `results/training_cost/train_metrics.csv`. `profile_sampling.py` measures the time to draw five synthetic datasets of training size (seeds 0 to 4) and an estimated parameter count, and writes `results/profiling/sampling_params.csv`. Both resume from their CSV; pass `--force` to recompute.

  ```bash
  python train_all_baselines.py --methods beta_copula gaussian_copula checkerboard_copula vine_copula
  python profile_sampling.py
  python profile_sampling.py --datasets adult cardio --methods beta_copula tabddpm
  ```

- **Paper tables and figures** — `analysis/run_all.py` recomputes every table, figure and number in the paper from the stored results (`results/benchmarks/benchmark_results.json`, `results/training_cost/train_metrics.csv` and `results/profiling/sampling_params.csv`). It ranks the seven ranked generators within each dataset, runs the Friedman test and the Nemenyi post hoc test, and writes LaTeX tables, CSVs and machine-readable facts to `results/paper/` and figures to `results/paper/figures/`.

  ```bash
  python analysis/run_all.py
  ```

  | Output | Paper |
  |---|---|
  | `table_p2_corr.tex` | Table 1 |
  | `table_ranks_means.tex` | Table 2 |
  | `table_cost.tex` | Table 3 |
  | `table_p2_std.tex` | Table C2 |
  | `table_ci_per_dataset.tex` | Table C3 |
  | `figures/cost_utility_pareto.pdf` | Figure 1 |
  | `figures/cd_diagrams.pdf` | Figure C1 |
  | `figures/train_time_vs_n.pdf` | Figure C2 |
  | `facts/*.json` | numbers quoted in the text (head-to-head counts, gaps to Real, cost ratios, log-log slopes) |

## Project Structure

```
Copula-Comparisons/
├── benchmark.py              # Main benchmark script
├── train_all_baselines.py    # Trains the deep baselines; records training time and model size
├── profile_sampling.py       # Records sampling time and parameter counts
├── beta_copula.py            # Empirical beta copula
├── checkerboard_copula.py    # Checkerboard copula
├── gaussian_copula.py        # Gaussian copula
├── vine_copula.py            # Vine copula (pyvinecopulib)
├── copula_utils.py           # One-hot decoding shared by the copulas
├── eval_metrics.py           # Fidelity, faithfulness, privacy and inference metrics
├── analysis/                 # Scripts that regenerate the paper's tables and figures
├── data/                     # Datasets (not tracked by git)
├── exp/                      # Baseline configs (trained models are not tracked)
│   └── {dataset}/
│       ├── ddpm_cb_best/     # TabDDPM
│       ├── tvae/             # TVAE
│       ├── ctabgan/          # CTAB-GAN
│       └── ctabgan-plus/     # CTAB-GAN+
├── results/
│   ├── benchmarks/           # Benchmark output (JSON and per-metric CSVs)
│   ├── training_cost/        # Training time and model size
│   ├── profiling/            # Sampling time and parameter counts
│   └── paper/                # Tables, figures and facts for the paper
├── tuned_models/             # Tuned CatBoost parameters
├── CTAB-GAN/                 # CTAB-GAN implementation
├── CTAB-GAN-Plus/            # CTAB-GAN+ implementation
├── CTGAN/                    # TVAE/CTGAN implementation
├── tab_ddpm/                 # TabDDPM implementation
├── scripts_tabddpm/          # TabDDPM pipeline scripts
└── lib/                      # Shared utilities (data loading, metrics, etc.)
```
