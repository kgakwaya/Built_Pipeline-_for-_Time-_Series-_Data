# Task 1 â€“ Time-Series Preprocessing, Exploratory Analysis and Forecasting (Household Electricity Consumption)


## 1. Overview and problem definition
Forecast a household's **average active power (`Global_active_power`, kW) one hour ahead** from hourly-aggregated history of the household's own measurements. Useful for household energy management, demand monitoring, efficiency baselines and short-term planning. The main deliverable is the executed notebook `notebooks/Task1_Time_Series_Preprocessing_EDA_Forecasting.ipynb`; the code in `src/` is imported and used by it.

## 2. Dataset

https://www.kaggle.com/datasets/sarahashmi/household-power-consumption-txt
* File: `household_power_consumption.txt` (from the supplied `archive (7).zip`). Semicolon separated, `?`/empty = missing, `Date` d/m/Y, `Time` H:M:S.
* Columns: `Date, Time, Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3`.
* Verified from the supplied file (computed in the notebook): 2,075,259 rows, 16 Dec 2006 17:24 â†’ 26 Nov 2010 21:02, strictly 1-minute grid with no absent timestamps, no duplicates, 25,979 rows (1.25 %) with all seven measurements missing in 71 runs (longest 5 days).
* **Provenance / units are not contained in the archive.** The dataset is commonly documented as the UCI "Individual household electric power consumption" dataset (units: kW, V, A; sub-metering in Wh per minute). This is a *source-dependent assumption*; verify and cite the original source yourself. No download link is given because none was supplied.

## 3. Prediction target and horizon
Target: hourly mean of `Global_active_power` at hour *T*. Forecast origin: end of hour *Tâˆ’1* (**horizon = 1 h, hourly frequency**). The modelling data are *hourly aggregates*, not minute-level data.

## 4. Project structure
```
task1_time_series/
â”œâ”€â”€ README.md  requirements.txt  .gitignore
â”œâ”€â”€ data/            raw/ (original file, untouched)  processed/ (generated)  README.md
â”œâ”€â”€ notebooks/       Task1_Time_Series_Preprocessing_EDA_Forecasting.ipynb
â”œâ”€â”€ src/             data_loading.py  preprocessing.py  feature_engineering.py  modeling.py
â”œâ”€â”€ outputs/         figures/  tables/  reports/  models/
â””â”€â”€ tests/           test_pipeline.py
```

## 5. Requirements, versions, installation
Validated with **Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.8, seaborn 0.13.2, joblib 1.5.3** (see `requirements.txt`; older compatible versions should work but were not tested).
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 6. Dataset placement
Put the original `household_power_consumption.txt` in `data/raw/` (or put the supplied zip in `data/raw/`; the notebook lists its contents and extracts only the expected member). See `data/README.md`. The raw file is never modified (SHA-256 checked at the end of the notebook).

## 7. Running
```bash
jupyter notebook notebooks/Task1_Time_Series_Preprocessing_EDA_Forecasting.ipynb   # then Kernel > Restart & Run All
# or non-interactively:
jupyter nbconvert --to notebook --execute --inplace notebooks/Task1_Time_Series_Preprocessing_EDA_Forecasting.ipynb
python -m unittest discover -s tests -v          # unit tests (pytest also works)
```
The notebook finds the project root itself, so it can be launched from the repository root or from `notebooks/`. Expected run time â‰ˆ 3â€“5 minutes on one CPU core (dominated by Random-Forest fitting/tuning).

## 8. Methodology summary
* **Preprocessing:** timestamp parsing and validation; duplicate policy (none found); missing data are explicit `?`/empty rows. Hourly **mean over observed minutes**; an hour is valid only with â‰¥ 45/60 observed minutes; sub-meter energy = mean Ã— 60 (Wh/h); causal forward-fill (â‰¤ 3 h) for *inputs only*, flagged; evaluated targets never imputed; longer gaps stay missing. A masking experiment compares fill strategies. Outliers kept (plausible peaks).
* **Analytical questions (Section 11):** Q1 trend; Q2 hour-of-day / weekday patterns; Q3 relationships with other measurements; **Q4 lagged features (ACF/PACF)**; **Q5 moving averages (smoothing, delay, leakage-safe shifted MAs)**; Q6 predictive usefulness of lag/MA/historical-mean predictors. Q4â€“Q6 use only the first 80 % of the timeline so the test period is never examined.
* **Features (31):** lags 1,2,3,6,12,24,48,168 of the target; rolling mean/SD over 3,6,24,168 h computed on the *shifted* series; lag-1 of the other measurements; calendar features of the *predicted* hour. Verified by assertions and a "corrupt-the-future" leakage test.
* **Split:** chronological 70/15/15 (train/validation/test); `TimeSeriesSplit` (4 folds) for tuning inside the training set only.
* **Models:** persistence and seasonal-naive baselines; Ridge; Random Forest; tuned versions (grid search for Ridge, randomised search for RF); ablation with target-only features. **Selection uses validation MAE only**; test metrics are reported afterwards.
* **Metrics:** MAE (primary), RMSE, RÂ²; MAPE (test, supplementary; only valid because actual values are > 0.05 kW).

## 9. Generated artifacts
`data/processed/` hourly datasets; `outputs/figures/` 17 PNG figures; `outputs/tables/` overview, missing-value report, decision table, descriptive statistics, feature dictionary, Q1â€“Q6 results, tuning results, `experiment_comparison.csv`, `final_test_predictions.csv`, `final_metrics.json`; `outputs/reports/results_summary.md`; `outputs/models/best_model.joblib` + `best_model_metadata.json`. Figures/tables are regenerated (overwritten) on every run.

## 10. Results (as produced by the executed notebook â€“ see `outputs/reports/results_summary.md`)
Chronological split: train 22,964 / validation 4,921 / test 4,922 hourly rows. Model chosen on **validation MAE**: **E4 (tuned Random Forest)**; test MAE 0.305 kW, RMSE 0.447 kW, RÂ² 0.600 vs persistence baseline MAE 0.386 kW, RÂ² 0.320. Interpret with care: one household, one test window (mean load 12 % lower than in training).

| experiment | model | Val_MAE | Val_RMSE | Val_R2 | Test_MAE | Test_RMSE | Test_R2 | runtime_s |
|---|---|---|---|---|---|---|---|---|
| E0a | Persistence (lag 1) | 0.4635 | 0.6915 | 0.3982 | 0.3855 | 0.5839 | 0.3195 | 0.0 |
| E0b | Seasonal naive 24 h | 0.6639 | 0.9602 | -0.1604 | 0.5089 | 0.7552 | -0.1385 | 0.0 |
| E0c | Seasonal naive 168 h | 0.5909 | 0.8723 | 0.0424 | 0.5499 | 0.7948 | -0.261 | 0.0 |
| E1 | Ridge (default) | 0.4075 | 0.5713 | 0.5892 | 0.3513 | 0.4909 | 0.5191 | 0.02 |
| E2 | Random Forest (default) | 0.353 | 0.5168 | 0.6638 | 0.3063 | 0.4487 | 0.5982 | 37.79 |
| E3 | Ridge (tuned) | 0.4076 | 0.5716 | 0.5888 | 0.3511 | 0.4906 | 0.5196 | 0.51 |
| E4 | Random Forest (tuned) | 0.3502 | 0.5164 | 0.6644 | 0.3046 | 0.4474 | 0.6004 | 100.94 |
| E5 | Ridge (tuned alpha), target-only features | 0.4128 | 0.585 | 0.5692 | 0.3503 | 0.496 | 0.509 | 0.01 |

## 11. Rubric compliance matrix (Task 1 only)
| Rubric requirement | Where (notebook section) | Evidence / output |
|---|---|---|
| Problem definition and dataset justification | Â§2 | Problem statement, applications, dataset rationale, target/horizon/unit table |
| Data exploration and preprocessing | Â§4â€“Â§9 | `dataset_overview.csv`, `missing_value_report.csv`, `descriptive_statistics_minute.csv`, `preprocessing_decision_table.csv`, figures 01â€“07 |
| Analytical questions (â‰¥ 5, each with plot + interpretation) | Â§11 Q1â€“Q6 | `analytical_question_results.csv`, figures 11â€“16 |
| Lagged-feature analysis | Â§11 Q4 | ACF/PACF table `Q4_lag_correlations.csv`, figure 14 |
| Moving-average analysis | Â§11 Q5 (+Q6) | `Q5_moving_average_results.csv`, `Q6_simple_predictors.csv`, figures 15â€“16 |
| Feature engineering | Â§12 | Leakage-safe lags/rolling/calendar, `feature_dictionary.csv`, assertions + leakage test |
| Model implementation | Â§14 | Baselines (E0aâ€“c), Ridge (E1/E3/E5), Random Forest (E2/E4) |
| Hyperparameter tuning | Â§15 | `GridSearchCV`/`RandomizedSearchCV` + `TimeSeriesSplit`; `tuning_*.csv` |
| Experimental design | Â§13, Â§17 | `split_summary.csv`, `experiment_comparison.csv` (validation-based selection) |
| Evaluation | Â§16â€“Â§18 | MAE/RMSE/RÂ²; figure 17; `final_test_predictions.csv`, `final_metrics.json` |
| Code quality | `src/`, `tests/` | Modular functions with docstrings; 21 unit tests |
| Documentation | this file, `data/README.md`, notebook Markdown | Setup, reproduction, limitations |



## 12. Troubleshooting
* *"Dataset not found"* â€“ place `household_power_consumption.txt` in `data/raw/`.
* *Project root not found* â€“ start Jupyter inside the repository.
* *Slow cell* â€“ Section 14â€“15 (Random Forest) dominates; lower `RF_N_ITER` in Section 3.
* *Pandas version errors* â€“ install the versions in Â§5.

## 13. Known limitations
Single household, historical data (2006â€“2010); hourly aggregation hides sub-hour peaks; rows with long gaps are excluded; one split / one test window with a shifted distribution; modest tuning space; predictive association â‰  causation; units from public documentation. Execution note: in the authoring environment no Jupyter kernel was installed, so the notebook was executed with a small custom runner that executes the code cells top-to-bottom in one namespace and records their real outputs; running it in Jupyter with the commands above has **not** been verified in that environment.
