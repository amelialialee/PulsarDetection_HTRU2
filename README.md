# HTRU2 Pulsar Detection 

A machine learning pipeline that classifies radio signal candidates from the HTRU2 survey as real pulsars or background noise, with a Streamlit app for interactive signal classification.

---

## Project structure

```
PulsarDetection_HTRU2/
├── data/
│   └── HTRU_2.csv  (Raw dataset)
├── models/
│   └── pulsar_pipeline.joblib  (Trained XGBoost pipeline)
├── reports/
│   ├── class_distribution.png
│   ├── correlation_matrix.png
│   ├── evaluation.png
│   ├── feature_distributions.png
│   ├── learning_curves.png
│   ├── shap_beeswarm_both_classes.png
│   ├── shap_force_nonpulsar_borderline.png
│   ├── shap_force_nonpulsar_confident.png
│   ├── shap_force_pulsar_borderline.png
│   ├── shap_force_pulsar_confident.png
│   └── shap_summary_bar_both_classes.png
├── src/
│   └── app.py  (Streamlit application)
└── notebooks/
    └── htru2_pulsar_classification.ipynb  (Full training & evaluation notebook)
```

---

## Dataset

**HTRU2** (High Time Resolution Universe Survey 2) is a benchmark dataset assembled from the Parkes radio telescope's survey. Each row is a labelled radio signal candidate described by 8 statistical features.

Total Candidates : 17,898

Real pulsars     : 1,639 (9.2%)

Background noise : 16,259 (90.8%) 

### Features

Each candidate is described by four statistical measures of two signal components:

| Feature | |
|---|---|
| Mean IP | Mean of the Integrated Profile |
| Std IP | Standard Deviation of the Integrated Profile |
| Kurt IP | Excess Kurtosis of the Integrated Profile |
| Skew IP | Skewness of the Integrated Profile |
| Mean DM | Mean of the DM-SNR Curve |
| Std DM | Standard Deviation of the DM-SNR Curve |
| Kurt DM | Excess Kurtosis of the DM-SNR Curve |
| Skew DM | Skewness of the DM-SNR Curve |

---

## Pipeline Decisions

### 1. Model shortlist
Four models were evaluated at default settings using identical 5 fold stratified cross-validation (SMOTE to StandardScaler to classifier):

| Model | F1 | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Dummy (always predicts non-pulsar) | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.846 | 0.907 | 0.793 | 0.977 |
| Random Forest | 0.867 | 0.896 | 0.841 | 0.976 |
| XGBoost | 0.809 | 0.908 | 0.731 | 0.975 |

Note: Logistic Regression was dropped as a linear model it cannot capture the non-linear feature interactions visible in the per-feature histograms. Random Forest and XGBoost were carried forward*

### 2. Imbalance handling
Three strategies were compared on Random Forest via 5-fold CV:

| Variant | F1 | Recall | Precision |
|---|---|---|---|
| SMOTE + class_weight='balanced' | 0.893 | 0.896 | 0.891 |
| SMOTE only | 0.896 | 0.901 | 0.892 |
| class_weight only (no SMOTE) | 0.893 | 0.897 | 0.890 |

SMOTE only won on recall and was marginally better across all metrics. Combining SMOTE with `class_weight='balanced'` was slightly redundant for tree models. SMOTE only was used as the base pipeline for all subsequent tuning.

### 3. Hyperparameter tuning
`RandomizedSearchCV` (35 iterations, 5-fold CV) was run for both RF and XGBoost, each tuned twice (once scoring on recall and once on F1).

The XGBoost recall-tuned variant was dropped: it achieved recall of 0.991 but precision collapsed to 0.167 (meaning 5 in 6 pulsar predictions would be false alarms, making it impractical)

### 4.Final model selection

| Model variant | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| RF (recall-tuned) | 0.901 | 0.826 | 0.862 | 0.979 |
| RF (F1-tuned) | 0.889 | 0.841 | 0.864 | 0.978 |
| **XGBoost (F1-tuned)** | 0.905 | 0.809 | 0.855 | 0.977 |

XGBoost (F1-tuned) was selected as it achieved the highest recall (0.905) at a practical precision (0.809). RF had a marginally higher ROC-AUC (0.979 vs 0.977) but this was outweighed by the recall advantage as recall was the primary metric throughout.

Nonetheless, RF remains a strong candidate for future work. Its higher ROC-AUC (0.979) suggests it may have more headroom under different tuning strategies (particularly  a deeper hyperparameter search or a lower classification threshold) and its recall-tuned variant came within 0.004 of XGBoost on recall. If precision becomes a higher priority in a future iteration, RF's better precision-recall balance at the recall-tuned setting makes it worth revisiting.

---

## Model performance

All metrics are from the held out test set (20% of HTRU2, untouched during training and tuning)

**Per-class breakdown** 
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-pulsar | 0.985 | 0.984 | 0.984 | 3,252 |
| Pulsar | 0.809 | 0.905 | 0.855 | 328 |
| Accuracy | | 0.974 | | 3,580 |
| Macro average | 0.897 | 0.945 | 0.920 | 3,580 |
| Weighted average | 0.975 | 0.974 | 0.974 | 3,580 |

**Model summary** 
| Metric | Score |
|---|---|
| ROC-AUC | 0.977 |
| PR-AUC | 0.929 |
| Train F1 | 0.927 |
| Test F1 | 0.855 |
| Overfitting gap | 0.073 |
| Nested CV F1 | 0.848 ± 0.006 |

### Tuned hyperparameters

| Parameter | Value | 
|---|---|
| n_estimators | 400 | 
| max_depth | 9 | 
| learning_rate | 0.01 | 
| subsample | 0.8 | 
| colsample_bytree | 0.7 |
| scale_pos_weight | 1 | 
---

## Application (Streamlit)

The application provides interactive signal classification with SHAP-based explanations.

**Install dependencies:**

```bash
pip install streamlit joblib shap numpy matplotlib pandas scikit-learn xgboost imbalanced-learn
```

**Run the app:**

```bash
streamlit run src/app.py
```

**Sections:**

- **Signal Classification**:  enter the 8 signal feature values and classify a signal. Returns a prediction, confidence score and SHAP  and a feature impact bar chart explaining the result for that specific signal.

- **Overview**: background and model information across three tabs
  - **WPulsar Overview**: Generally what pulsars are, how they are detected, and what this project is trying to do
  - **Model Overview**: The XGBoost model, performance metrics, tuned hyperparameters, primary metric reasoning and the HTRU2 dataset.
  - **Feature Guide**: What each of the 8 signal features measures, how the two signal components (Integrated Profile and DM-SNR Curve) work, and what distinguishes pulsar signals from background noise feature by feature.
---

## Notebook/Pipeline

`notebooks/htru2_pulsar_classification.ipynb` contains the full end-to-end pipeline:

1. Load and split HTRU_2.csv (stratified 80/20)
2. EDA (class distribution, feature histograms, correlation matrix)
3. 5-fold CV model comparison (Dummy, Logistic, RF, XGBoost)
4. Learning curves (RF and XGBoost)
5. Ablation study (imbalance handling strategies)
6. Hyperparameter tuning (RandomizedSearchCV for RF and XGBoost)
7. Head-to-head test-set comparison
8. Nested CV sanity check
9. Final evaluation (classification report, ROC, PR curve)
10. SHAP interpretability (bar, beeswarm, and force plots)
11. Save model to `models/pulsar_pipeline.joblib`

---

## Requirements

```
numpy==2.3.5
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.6
seaborn==0.13.2
joblib==1.5.2
shap==0.47.2
xgboost==3.0.2
streamlit==1.51.0
imbalanced-learn==0.14.0
ipykernel==6.31.0
```

---

## reference

R. J. Lyon et al., *Fifty Years of Pulsar Candidate Selection*, MNRAS, 2016.  
HTRU2 dataset via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/372/htru2).