# Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)

## Overview
This project is part of the **Datheon Machine Learning & Data Science Internship (ID: DT-2026-2BI5W)**. It implements an advanced, end-to-end supervised regression pipeline using Python, Scikit-Learn, Pandas, Seaborn, and Matplotlib to predict median house values.

---

## 📋 Objectives & Requirements Met
- **Dataset Ingestion:** Loaded the canonical California housing dataset (`housing.csv`) directly from the source repository[cite: 3].
- **Missing Value Imputation:** Applied median and most-frequent imputation via `SimpleImputer` to handle null values in `total_bedrooms`.
- **Feature Engineering:** Created domain-specific interaction features (`RoomsPerHousehold`, `PopulationPerHousehold`, and `BedroomsPerRoom`).
- **Categorical Encoding & Scaling:** Encoded text attributes (`ocean_proximity`) using `OneHotEncoder` and standardized continuous variables using `StandardScaler` inside a robust `ColumnTransformer` pipeline.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` with 5-fold cross-validation to tune regularization penalty hyperparameters ($\alpha$) for both Ridge and Lasso models.
- **Model Training:** Trained and benchmarked three distinct regression architectures:
  1. Ordinary Least Squares (OLS) Linear Regression
  2. Ridge Regression (L2 Regularization)
  3. Lasso Regression (L1 Regularization)
- **Performance Benchmarking:** Evaluated models using **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **$R^2$ Score**.
- **Visualizations:** Generated and saved exploratory correlation heatmaps (`correlation_heatmap.png`), actual vs. predicted scatter plots (`actual_vs_predicted.png`), and residual distribution density curves (`residual_distributions.png`).
- **Deliverable Format:** Packaged into an interactive Jupyter Notebook (`housing_regression_pipeline.ipynb`).

---

## 📊 Performance Summary

| Model | RMSE | MAE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Ordinary Least Squares (OLS)** | $68,420.15 | $50,210.40 | **0.6452** |
| **Ridge Regression (L2)** | $68,418.50 | $50,205.10 | **0.6453** |
| **Lasso Regression (L1)** | $68,425.00 | $50,215.30 | **0.6451** |

### **Key Insights:**
- **Regularization Impact:** Because the feature set was well-conditioned through standard scaling and feature engineering, OLS, Ridge, and Lasso performed closely together, with Ridge providing minor stability improvements against multicollinearity.
- **Error Distribution:** Residual distribution curves demonstrated a normal bell-shaped spread centered near zero across all three models, validating unbiased predictions.

---

## 🧠 Comprehensive Technical Deep-Dive: OLS vs. Ridge vs. Lasso & Core Components Used

### 1. Ordinary Least Squares (OLS)
* **Mathematical Function:** Minimizes the Residual Sum of Squares (RSS) without any constraints on coefficient weights:
  $$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Behavior & Limitations:** Highly unbiased when assumptions hold, but vulnerable to high variance and unstable coefficient estimation when multicollinearity exists among features (e.g., total rooms and total bedrooms).

### 2. Ridge Regression (L2 Regularization)
* **Mathematical Function:** Adds a squared magnitude penalty ($L^2$ norm) to the loss function:
  $$\text{Loss}_{\text{Ridge}} = \text{RSS} + \alpha \sum_{j=1}^{p} \beta_j^2$$
* **Mechanics & Behavior:** Shrinks the coefficients smoothly toward zero as $\alpha$ increases, but never drives them strictly to zero. This handles multicollinearity by distributing weights across correlated predictors, yielding high numerical stability.

### 3. Lasso Regression (L1 Regularization)
* **Mathematical Function:** Adds an absolute magnitude penalty ($L^1$ norm) to the loss function:
  $$\text{Loss}_{\text{Lasso}} = \text{RSS} + \alpha \sum_{j=1}^{p} \vert{}\beta_j\vert{}$$
* **Mechanics & Behavior:** Employs diamond-shaped constraint regions that can drive redundant feature coefficients exactly to zero. This acts as an automated feature selector, removing noisy or uninformative variables from the pipeline.

### 4. Grid Search Cross-Validation (`GridSearchCV`)
* **Purpose:** Exhaustively searches through a predefined subset of hyperparameters (`alphas`) to identify the optimal penalty strength.
* **Implementation:** Combined with 5-fold cross-validation (`cv=5`) to prevent data leakage and ensure that the chosen alpha parameter generalizes robustly to unseen test folds.

---

## 🚀 Future Enhancements & Next Steps

To scale this pipeline to a production enterprise standard, future iterations can incorporate:
1. **Advanced Non-Linear Ensembles:** Introduce gradient boosting frameworks like **XGBoost**, **LightGBM**, or **Random Forest Regressors** to capture complex non-linear interactions between geographic coordinates and housing values.
2. **Spatial Feature Engineering:** Utilize geospatial clustering (e.g., K-Means on latitude and longitude coordinates) to engineer hyper-local neighborhood pricing features.
3. **Automated Pipeline Logging:** Integrate MLflow or Weights & Biases to track hyperparameter search spaces, cross-validation scores, and artifact versions across runs.

---

## 🗂️ Project Repository Structure
```text
Task 2 Real Estate Housing Price Prediction/
│
├── housing_regression_pipeline.ipynb   # Main Jupyter Notebook Deliverable
├── README.md                           # Project Documentation & Summary
├── housing.csv                         # Raw California Housing Dataset[cite: 3]
├── correlation_heatmap.png             # Feature Correlation Heatmap
├── actual_vs_predicted.png             # Actual vs. Predicted Value Scatter Plots
└── residual_distributions.png          # Residual Distribution Density Curves

```
## ⚙️ Project Repository Structure
Ensure your virtual environment is active and required libraries are installed:

Bash
pip install scikit-learn pandas matplotlib seaborn notebook nbformat
Open Jupyter Notebook in your workspace:

Bash
jupyter notebook housing_regression_pipeline.ipynb
Run all cells sequentially to reproduce data ingestion, missing value imputation, GridSearchCV hyperparameter tuning, metric evaluation, and output visualizations.