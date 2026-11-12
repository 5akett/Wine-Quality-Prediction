# 🍷 Wine Quality Prediction — CSE570 Machine Learning Project

**End-to-End Machine Learning Solution for Predicting Red Wine Quality from Physicochemical Properties**

**Course:** CSE570 — Machine Learning with Python  
**Academic Task:** Skill Based Assignment  
**Student:** Ritik Kumar  
**Dataset:** [UCI Red Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)

---

## 📋 Problem Statement

A Portuguese wine company (Vinho Verde) wants to predict the quality of red wine based on physicochemical lab test results. Manual tasting by experts is expensive and slow. By training a machine learning model on chemical features, winemakers can estimate quality at scale, reduce tasting costs, and make informed decisions about batch pricing and production adjustments.

- **Target Variable:** `quality` — expert score from 3 to 8
- **ML Task:** Regression (continuous prediction)
- **Success Criteria:** R² > 0.30 and RMSE < 0.70 on test data
- **Impact:** Reduce manual tasting costs, assist in production decisions and batch pricing

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) |
| Records | 1,599 (1,358 after deduplication) |
| Features | 11 physicochemical properties |
| Target | `quality` (3–8, expert rating) |
| Missing Values | None |
| Citation | Cortez, P. et al. (2009). *Decision Support Systems*, 47(4), 547–553 |

### Features

| Feature | Description |
|---|---|
| `fixed acidity` | Most acids involved with wine (g/dm³) |
| `volatile acidity` | Acetic acid — high levels cause vinegar taste |
| `citric acid` | Adds freshness and flavor |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt amount in wine |
| `free sulfur dioxide` | Prevents microbial growth |
| `total sulfur dioxide` | Free + bound SO₂ |
| `density` | Density of wine (close to water) |
| `pH` | Acidity/basicity on scale 0–14 |
| `sulphates` | Wine additive (antioxidant) |
| `alcohol` | Alcohol percentage (%) |
| `quality` | **Target:** expert score 3–8 |

---

## 🧪 Methodology

```
Data Loading → Cleaning → EDA → Feature Scaling → Model Training → Evaluation → Comparison
```

1. **Data Loading** — Load from UCI repository URL (semicolon-separated CSV)
2. **Data Cleaning** — Duplicate removal (241 rows), missing-value check, outlier inspection via IQR
3. **EDA** — Univariate distributions, outlier boxplots, correlation heatmap, bivariate scatter plots, multivariate quality-grouped plots
4. **Preprocessing** — StandardScaler (fit on train only, no data leakage)
5. **Modeling** — Simple LR, Multiple LR, Polynomial (deg=2), Ridge (L2), Lasso (L1), Elastic Net (L1+L2)
6. **Evaluation** — MAE, MSE, RMSE, R² metrics + actual-vs-predicted visualizations
7. **Bonus** — Classification view: Good (≥7) vs Bad (≤6) using Logistic Regression, KNN, Random Forest

---

## 📁 Project Structure

```
Wine Quality Prediction/
├── 01_Introduction_Data_Loading.ipynb     # Problem statement, dataset, imports, load & inspect
├── 02_Data_Cleaning_Preparation.ipynb     # Duplicates, missing values, train-test split, scaling
├── 03_EDA_Statistical_Analysis.ipynb      # Univariate, outliers, correlation, bivariate, multivariate
├── 04_Regression_Models.ipynb             # Simple LR, Multiple LR, Polynomial, Ridge, Lasso, ElasticNet
├── 05_Model_Comparison_Classification.ipynb  # Model comparison, actual vs predicted, classification
├── 06_Conclusion_References.ipynb         # Conclusion, key findings, limitations, references
├── Wine_Quality_Prediction_CSE570.ipynb   # Full single notebook (all cells combined)
├── README.md
└── requirements.txt
```

---

## 🔑 Key Findings

1. **Alcohol** is the strongest positive predictor of wine quality (r = 0.48)
2. **Volatile acidity** is the strongest negative predictor (r = -0.39)
3. **Citric acid** and **sulphates** also contribute positively
4. **~13%** of wines are rated "excellent" (7–8); most are average (5–6)
5. **Best model:** Multiple Linear Regression or Ridge with R² ≈ 0.35, RMSE ≈ 0.62

---

## ⚙️ Setup & Execution

### Prerequisites
- Python 3.8+
- Jupyter Notebook / JupyterLab

### Installation

```bash
git clone https://github.com/<your-username>/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook 01_Introduction_Data_Loading.ipynb
```

Or run all notebooks sequentially:

```bash
jupyter notebook
```

> **Note:** No manual dataset download is required — the notebooks load the CSV directly from the UCI repository URL.

---

## 📈 Results Summary

| Model | Best Hyperparameters | 5-Fold CV R² | Test R² | RMSE | MAE |
|---|---|---|---|---|---|
| Simple Linear Regression | — | — | ~0.33 | ~0.64 | ~0.50 |
| Multiple Linear Regression | — | — | ~0.36 | ~0.62 | ~0.48 |
| Polynomial (deg=2) | — | — | ~0.36 | ~0.62 | ~0.47 |
| Ridge (L2) | alpha=100.0 | 0.3225 | 0.3957 | 0.6542 | 0.5048 |
| Lasso (L1) | alpha=0.01 | 0.3259 | 0.3995 | 0.6522 | 0.5012 |
| ElasticNet (L1+L2) | alpha=0.01, l1_ratio=0.7 | 0.3263 | 0.3989 | 0.6525 | 0.5015 |
| **Random Forest (SELECTED)** | **max_depth=10, n_estimators=200** | **0.3442** | **0.4642** | **0.6160** | **0.4689** |

**Best Model:** Random Forest (max_depth=10, n_estimators=200) — selected by highest 5-fold CV R² (0.3442). Test R² = 0.4642, RMSE = 0.6160, MAE = 0.4689. It beats all linear/regularized models, showing the quality relationship is non-linear.

The final model and scaler are saved to disk and can be reused:

```python
import joblib, pandas as pd

model = joblib.load('wine_quality_model.joblib')
scaler = joblib.load('wine_quality_scaler.joblib')

new_wine = pd.DataFrame([{
    'fixed acidity': 7.4, 'volatile acidity': 0.70, 'citric acid': 0.0,
    'residual sugar': 1.9, 'chlorides': 0.076, 'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0, 'density': 0.9978, 'pH': 3.51,
    'sulphates': 0.56, 'alcohol': 9.4
}])
predicted_quality = model.predict(scaler.transform(new_wine))
print(predicted_quality)
```

### Saved Model Files

Each tuned model is persisted individually, plus the selected best model and the fitted scaler:

| File | Description |
|---|---|
| `wine_quality_ridge.joblib` | Ridge regression (alpha=100.0) |
| `wine_quality_lasso.joblib` | Lasso regression (alpha=0.01) |
| `wine_quality_elasticnet.joblib` | ElasticNet (alpha=0.01, l1_ratio=0.7) |
| `wine_quality_randomforest.joblib` | Random Forest (max_depth=10, n_estimators=200) |
| `wine_quality_model.joblib` | **Selected best model** (Random Forest) |
| `wine_quality_scaler.joblib` | Fitted StandardScaler (used to scale inputs) |
| `model_comparison.csv` | Full comparison table of all models |

**Always apply `wine_quality_scaler.joblib` to new inputs before calling `predict`** — the models were trained on standardized features.

---

## ⚠️ Limitations

- R² ≈ 0.35 means ~65% of variance is unexplained — quality depends on factors beyond chemistry (taste is subjective)
- Polynomial regression may overfit despite better training performance
- Cross-validation and hyperparameter tuning (GridSearchCV) could improve regularization
- Dataset is from a single region (Minho, Portugal) — generalizability to other wine regions is unverified

---

## 📚 References

1. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547–553.
2. UCI Machine Learning Repository: [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
3. Scikit-learn Documentation: https://scikit-learn.org/stable/
4. Python Data Science Handbook — Jake VanderPlas (O'Reilly, 2016)

---

## 📄 License

This project is for academic purposes only. Dataset is publicly available under the UCI ML Repository license.

---

## 🙋 Author

**Ritik Kumar** — CSE570 Machine Learning with Python  
[GitHub](https://github.com/ritik) | [LinkedIn](https://linkedin.com/in/ritik)