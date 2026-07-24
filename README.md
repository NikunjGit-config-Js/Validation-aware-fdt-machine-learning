# 📄 Cross-Validated Machine Learning Prediction of the In-Vitro Parameters of Fast-Disintegrating Tablets

### A Four-Protocol Comparative Study

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-006600)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-02569B)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-50%20Trials-2C3E50)](https://optuna.org/)
[![IIT BHU](https://img.shields.io/badge/IIT%20(BHU)-Varanasi-7A1F2B)](https://www.iitbhu.ac.in/)

**A validation-engineering thesis that treats model tuning and evaluation protocol as independent experimental factors.** Seventeen regression algorithms are evaluated across a controlled 2×2 design to determine which published pharmaceutical-ML results remain stable under stronger validation—and which are artefacts of one favourable random split.

| Academic metadata | Details |
|---|---|
| **Author** | Nikunj Mathur |
| **Roll number** | 22164011 |
| **Programme** | Integrated Dual Degree in Pharmaceutical Engineering and Technology |
| **Institution** | Indian Institute of Technology (BHU), Varanasi |
| **Supervisor** | Dr. Ashish Kumar Agarwal |
| **Dataset** | 1,982 formulations × 79 columns; literature compiled from 2010–2020 |
| **Primary contribution** | 17 models × 3 targets × 4 protocols = **204 evaluated model configurations** |

The project predicts three in-vitro properties of fast-disintegrating tablets:

- **Disintegration time**
- **Friability**
- **Water absorption ratio**

The dataset is available from the [Harvard Dataverse](https://doi.org/10.7910/DVN/TUSJYB). Results are compared against [Gupta et al. (2024)](https://doi.org/10.1016/j.ejpb.2024.114508), which used the same dataset.

## 🚀 The 4-Protocol Experimental Design

The experiment is a **2×2 factorial engineering comparison**:

| Validation protocol ↓ / Model configuration → | Default hyperparameters | Optuna-tuned hyperparameters |
|---|---|---|
| **Single 80:20 hold-out** | **Test 1:** baseline hold-out | **Test 2:** hold-out + Optuna, 50 trials |
| **5-fold cross-validation** | **Test 3:** 5-fold CV baseline | **Test 4:** 5-fold CV + Optuna, 50 trials |

### What each controlled comparison isolates

| Comparison | Variable held constant | Effect isolated |
|---|---|---|
| **Test 1 vs Test 3** | Default model parameters | **Validation protocol:** one 80:20 split versus five-fold CV |
| **Test 1 vs Test 2** | Same 80:20 split | **Hyperparameter tuning:** defaults versus Optuna |
| **Test 3 vs Test 4** | Same five-fold partitioning | **Hyperparameter tuning under CV:** defaults versus Optuna |
| **Test 1 vs Test 4** | Same cleaned dataset and target definitions | Full change from the least rigorous to the most rigorous configuration |

For Tests 2 and 4, every tunable model receives **50 Optuna trials per target** using a seeded Tree-structured Parzen Estimator and a median pruner. Test 2 tunes only inside the training partition; its 20% test partition remains untouched until final evaluation.

The 17-model search space contains:

`Linear Regression` · `Ridge` · `Lasso` · `Elastic Net` · `Decision Tree` · `Random Forest` · `Extra Trees` · `Gradient Boosting` · `Histogram Gradient Boosting` · `AdaBoost` · `Bagging` · `XGBoost` · `LightGBM` · `SVR` · `KNN` · `Voting Regressor` · `Stacking Regressor`

Every configuration is assessed using **MSE, MAE, RMSE, R² and adjusted R²**. Cross-validation experiments also retain fold-level dispersion to expose unstable results that a single aggregate score would hide.

## 🛠️ Repository Reproducibility Pathway

The five notebooks form a one-producer, four-consumer workflow:

```text
Harvard Dataverse raw CSV (read-only)
                │
                ▼
Notebook 1: EDA + pharmaceutical anomaly detection
                │
                └── writes preprocessed.csv exactly once
                              │
             ┌────────────────┼────────────────┬────────────────┐
             ▼                ▼                ▼                ▼
          Test 1           Test 2           Test 3           Test 4
          Hold-out         Hold-out         5-fold CV        5-fold CV
          Defaults         + Optuna         Defaults         + Optuna
```

`Notebook 1` is the **only component permitted to write `preprocessed.csv`**. Each of the four test notebooks reads that same file independently; no test notebook consumes another test's fitted estimator or in-memory state. This prevents preprocessing drift from becoming a hidden experimental variable.

All notebooks use:

```python
SEED = 42
```

This seed controls data splitting, shuffled folds, supported estimators and Optuna's TPE sampler.

### Environment setup with Conda

```bash
git clone https://github.com/NikunjGit-config-Js/<repository-name>.git
cd <repository-name>

conda create --name fdt-validation python=3.10 -y
conda activate fdt-validation

python -m pip install --upgrade pip
python -m pip install \
  numpy pandas matplotlib seaborn jupyter nbconvert \
  scikit-learn xgboost lightgbm optuna
```

### Execution order

The notebooks were originally developed in Google Colab and contain `/content/` path constants. For local execution, replace those constants with the repository-relative paths shown below:

```python
from pathlib import Path

RAW_CSV = Path("../data/raw/final_Data_All_Excipients.csv")
PREPROCESSED_CSV = Path("../data/processed/preprocessed.csv")
RESULTS_DIR = Path("../results")
FIGURES_DIR = Path("../figures")
```

Then execute the notebooks from the `notebooks/` directory:

```bash
mkdir -p executed_notebooks
cd notebooks

# Producer: creates data/processed/preprocessed.csv
jupyter nbconvert --to notebook --execute \
  01_EDA_and_Anomaly_Detection.ipynb \
  --output-dir ../executed_notebooks \
  --ExecutePreprocessor.timeout=-1

# Independent consumers: Tests 1–4
jupyter nbconvert --to notebook --execute \
  02_Test_1_Baseline_Holdout.ipynb \
  --output-dir ../executed_notebooks \
  --ExecutePreprocessor.timeout=-1

jupyter nbconvert --to notebook --execute \
  03_Test_2_Holdout_Optuna.ipynb \
  --output-dir ../executed_notebooks \
  --ExecutePreprocessor.timeout=-1

jupyter nbconvert --to notebook --execute \
  04_Test_3_FiveFold_CV.ipynb \
  --output-dir ../executed_notebooks \
  --ExecutePreprocessor.timeout=-1

jupyter nbconvert --to notebook --execute \
  05_Test_4_FiveFold_CV_Optuna.ipynb \
  --output-dir ../executed_notebooks \
  --ExecutePreprocessor.timeout=-1
```

Tests 1–4 may be fitted independently once `preprocessed.csv` exists. Run Tests 1–3 before the final comparison cell in Test 4 because that cell aggregates the four result CSVs. Approximate Colab wall times were **52 minutes for Test 2** and **4.2 hours for Test 4**.

## 📊 Data Engineering & Pharmaceutical Physics Cleaning

The raw 1,982 × 79 Dataverse file is opened read-only and never overwritten. Cleaning is based on **pharmaceutical powder physics and definitional consistency**, not generic z-score or percentile filtering.

### Rule 1 — Density physics

For a physically consistent powder:

```text
0 ≤ Bulk Density ≤ Tapped Density ≤ 1.5 g/mL
```

Tapped density cannot be below bulk density because tapping settles an already-loose powder bed. Neither density should exceed 1.5 g/mL in this dataset's pharmaceutical context. This rule caught, among other values, a source entry of **840 g/mL**—a clear decimal or transcription error rather than a plausible powder measurement.

When either density in a row violated the rule, both density cells were invalidated together so that downstream calculations could not mix one credible value with one physically impossible value.

### Rule 2 — Definitional consistency

Hausner ratio and Carr's compressibility index are not independent observations:

```math
Hausner\ Ratio = \frac{\rho_{tapped}}{\rho_{bulk}}
```

```math
Carr's\ Index = 100 \times \frac{\rho_{tapped}-\rho_{bulk}}{\rho_{tapped}}
```

The pipeline flags a reported Hausner ratio or Carr's Index when it disagrees with the value implied by its own source densities by more than **5%**.

### Rule 3 — Sign constraints without destroying zeros

Disintegration time, friability, thickness, wetting time, water absorption ratio, bulk density and tapped density cannot be negative. Negative entries are invalidated, but **exact zeros are retained as recorded data**. Treating all zeros as missing would silently change the prediction task—especially for water absorption, where more than half of the target is zero.

### Repair accounting and leakage control

| Anomaly class | Cells set to NaN |
|---|---:|
| Impossible bulk/tapped densities | 60 |
| Carr's Index inconsistent with source densities | 359 |
| Hausner ratio inconsistent with source densities | 155 |
| **Total repaired cells** | **574** |

The 574 cells represent **0.37% of the dataset's 156,578 cells**. No row is deleted during the primary cleaning stage, and no invalid value is replaced with a guessed number.

Missing-value handling is delegated to a median `SimpleImputer` inside each scikit-learn `Pipeline`. During five-fold CV, the imputer—and `RobustScaler` for scale-sensitive models—is refitted only on the current training fold. The validation fold therefore cannot influence imputation medians or scaling statistics.

## 📈 Critical Findings & Results Matrix

The matrix below reports **R² / RMSE** for the best model under each primary protocol. Tests 3 and 4 use the mean of the five fold-level metrics. Qualified sub-population analyses are shown separately because they answer different scientific questions.

| Target | Gupta et al. (2024) | Test 1: Hold-out | Test 2: Hold-out + Optuna | Test 3: 5-fold CV | Test 4: 5-fold CV + Optuna |
|---|---|---|---|---|---|
| **Disintegration time** | Voting: **0.760 / 21.989 s** | Extra Trees: **0.736 / 21.925 s** | LightGBM: **0.752 / 21.248 s** | Extra Trees: **0.755 / 21.924 s** | Extra Trees: **0.763 ± 0.057 / 21.587 s** |
| **Friability, all rows** | Random Forest: **0.700 / 0.142*** | AdaBoost: **0.697 / 0.540** | AdaBoost: **0.646 / 0.583** | SVR: **0.047 / 0.930** | SVR: **0.133 ± 0.153 / 0.899** |
| **Water absorption ratio** | KNN: **0.940 / 10.073** | Extra Trees: **0.942 / 10.696** | Extra Trees: **0.942 / 10.683** | Extra Trees: **0.945 / 9.676** | Extra Trees: **0.947 ± 0.018 / 9.554** |

\* The reference friability score excludes rows above 2% friability. It is therefore not directly comparable with the untreated all-row Test 1–4 entries until the same scoring population is used.

> **Metric convention:** fold-mean scores are used in the table to match the reference study's reporting convention. Actual-versus-predicted CV figures use pooled out-of-fold predictions, so their R² and RMSE can differ slightly without representing a contradiction.

![Best-model progression across the four protocols](figures/fig_progression.png)

### Breakthrough finding 1 — Friability is protocol-dependent

The apparently strong hold-out result, **R² = 0.697**, collapsed to **R² = 0.047** when the same default-model comparison was evaluated with five-fold CV. This was not a tuning failure. It was a target-distribution and evaluation-protocol failure.

Only **7 of 1,982 formulations (0.35%)** have friability above 2%. Those seven observations carry almost the entire variance of the target:

- In the favourable hold-out split, extreme rows make AdaBoost appear highly predictive.
- Across five folds, each rare extreme row creates a large, unstable error in whichever fold contains it.
- Fold-level variance exposes this instability; one hold-out score conceals it.

A like-for-like sensitivity analysis applied the reference study's exact scoring rule by excluding the seven rows above 2%. Under five-fold CV, Extra Trees then achieved:

| Friability scoring population | R² | RMSE |
|---|---:|---:|
| All 1,982 rows | 0.185 | 0.913 |
| Rows ≤ 2% friability | **0.693** | **0.144** |
| Gupta et al. reference | **0.700** | **0.142** |

The reference result is therefore reproducible under stronger validation—but only inside the normal pharmaceutical range. The analysis proves that the published friability performance is **population- and protocol-contingent**, not universally valid for catastrophic tablet failure.

![Friability sensitivity analysis](figures/fig_friability_sensitivity.png)

### Breakthrough finding 2 — Water absorption is zero-inflated

Exactly **1,046 of 1,982 rows (52.8%)** are recorded with a water absorption ratio of zero; only **936 rows** carry a non-zero measurement.

The tuned Extra Trees model separates the zero and measured populations extremely well, producing the headline all-row result:

| Water-absorption scoring population | R² | RMSE |
|---|---:|---:|
| All 1,982 rows | **0.947** | **9.554** |
| 936 measured rows only | **0.853** | **13.402** |

The first score describes the combined zero-separation and regression task. The second is the honest estimate of continuous regression performance among formulations with an actual recorded measurement.

![Water-absorption zero-inflation analysis](figures/fig_water_zero_inflation.png)

### Why Extra Trees dominated the stable targets

The dataset is **wide (76 predictors), moderately sized (1,982 rows), noisy and literature-compiled**. Extra Trees randomises both the feature subset and the split threshold, aggressively reducing estimator variance. That behaviour is well matched to many weak, interacting predictors and heterogeneous laboratory measurements.

Consequently, Extra Trees was the best model for:

- Disintegration time in three of four protocols.
- Water absorption ratio in all four protocols.
- The like-for-like friability sensitivity analysis after applying the reference study's valid-range scoring rule.

## 💾 Model Artifacts, Limitations & Future Work

### Repository directory structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   │   └── final_Data_All_Excipients.csv
│   └── processed/
│       └── preprocessed.csv
├── notebooks/
│   ├── 01_EDA_and_Anomaly_Detection.ipynb
│   ├── 02_Test_1_Baseline_Holdout.ipynb
│   ├── 03_Test_2_Holdout_Optuna.ipynb
│   ├── 04_Test_3_FiveFold_CV.ipynb
│   └── 05_Test_4_FiveFold_CV_Optuna.ipynb
├── results/
│   ├── results_test1_baseline.csv
│   ├── results_test2_optuna.csv
│   ├── results_test2_optuna_trials.csv
│   ├── results_test3_cv.csv
│   ├── results_test4_cv_optuna.csv
│   ├── predictions_all_tests.csv
│   ├── BEST_overall_model.csv
│   └── COMPARISON_vs_paper.csv
├── figures/
│   ├── fig_pipeline.png
│   ├── fig_progression.png
│   ├── fig_friability_sensitivity.png
│   ├── fig_water_zero_inflation.png
│   └── fig_vs_reference.png
└── reports/
    ├── FDT_Thesis_Nikunj_Mathur.pdf
    └── FDT_Explanation_and_Defence_Manual.pdf
```

The repository's primary artifacts are deterministic result tables, Optuna trial histories, pooled out-of-fold predictions and diagnostic figures. Serialized estimators are intentionally not presented as universally deployable models: their valid domain depends on the target population and validation scope documented below.

### Limitations

1. **This is not a pure a priori formulation model.** Wetting time, hardness, thickness and drug content are measured after compression. Wetting time is particularly informative. The practical use case is therefore predicting slower or more expensive tablet measurements from cheaper post-compression measurements—not predicting every property from formulation composition alone.

2. **The validation scope is chemical interpolation, not true extrapolation.** Random folds usually place formulations of the same or chemically related active pharmaceutical ingredient in both training and validation data. The defensible claim is performance on new formulations of represented drugs. Performance on a genuinely unseen drug requires grouping by active ingredient with `GroupKFold`.

### Future work

- **Two-stage hurdle model for water absorption:** first classify zero versus measured, then regress only on measured rows.
- **Post-compression ablation study:** remove wetting time, hardness, thickness and drug content to quantify fully a priori predictive performance.
- **SHAP analysis of tuned Extra Trees:** measure global and formulation-level feature contributions while checking whether wetting-time dominance persists.

---

**Nikunj Mathur** · Integrated Dual Degree, Pharmaceutical Engineering and Technology · IIT (BHU), Varanasi  
**Supervisor:** Dr. Ashish Kumar Agarwal
