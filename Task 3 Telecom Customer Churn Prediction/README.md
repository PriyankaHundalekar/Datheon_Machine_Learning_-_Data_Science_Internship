# Task 3: Telecom Customer Churn Prediction & Ensemble Modeling

## Overview
This project is part of the **Datheon Machine Learning & Data Science Internship (ID: DT-2026-2BI5W)**. It delivers a production-grade, end-to-end machine learning pipeline to predict customer churn using telecom subscription records from IBM's Telco Customer Churn dataset.

The workflow addresses real-world machine learning challenges, including missing data imputation, categorical encoding via Scikit-Learn's `OneHotEncoder`, class imbalance mitigation using **SMOTE** (Synthetic Minority Over-sampling Technique), multi-model ensemble training (**Random Forest** and **XGBoost**), cross-validation, and model explainability via feature importance ranking and **ROC-AUC analysis**.

---

## 📋 Objectives & Requirements Met
- **Dataset Ingestion & Cleaning:** Loaded the Telco Customer Churn dataset from IBM's open repository, converted `TotalCharges` to numeric data types, and filled null values using median imputation.
- **Categorical Feature Preprocessing:** Utilized Scikit-Learn's `ColumnTransformer` with `OneHotEncoder(drop='first', handle_unknown='ignore')` alongside `StandardScaler` for continuous features to prevent data leakage.
- **Class Imbalance Mitigation:** Applied **SMOTE** on the training split to synthesize minority churn class samples, balancing the target distribution.
- **Ensemble Classifier Training:** Trained and cross-validated two powerful tree-based ensemble models:
  1. `RandomForestClassifier`
  2. `XGBClassifier`
- **Performance Evaluation:** Evaluated models using 5-Fold Stratified Cross-Validation, out-of-sample confusion matrices, classification reports, and ROC-AUC curve visualizations (exceeding the required **>80% ROC-AUC** benchmark threshold).
- **Explainability & Feature Importance:** Extracted and visualized the Top 10 features driving customer churn using Random Forest impurity-based feature importances.
- **Deliverable Format:** Fully documented interactive Jupyter Notebook (`churn_prediction_pipeline.ipynb`) and structured repository documentation.

---

## 📊 Performance Summary

| Model | 5-Fold CV Mean ROC-AUC | Test ROC-AUC | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **91.85%** | **83.12%** | **78.21%** | **78.64%** | **78.21%** | **78.40%** |
| **Random Forest Classifier** | **92.30%** | **82.45%** | **77.50%** | **78.10%** | **77.50%** | **77.76%** |

*(Note: Test set performance evaluated on the original 20% unseen test split following SMOTE training).*

### **Key Insights:**
- **Benchmark Target Met:** Both ensemble algorithms cleanly surpassed the **80% ROC-AUC requirement**, with XGBoost reaching an out-of-sample test ROC-AUC of **~83.1%** and Random Forest hitting **~82.5%**.
- **Impact of SMOTE:** Balancing the training set significantly improved minority class (churners) recall, enabling early identification of high-risk customers without sacrificing overall decision boundaries.
- **Top Churn Drivers:** Feature importance analysis revealed that **Tenure**, **TotalCharges**, **MonthlyCharges**, **Contract Type (Month-to-month)**, and **Internet Service Type (Fiber Optic)** are the strongest predictors of customer attrition.

---

## 🗂️ Project Repository Structure
```text
Task 3 Telecom Customer Churn Prediction/
│
├── churn_prediction_pipeline.ipynb      # Main Jupyter Notebook Pipeline
├── README.md                            # Comprehensive Project Documentation
├── churn_roc_auc_curves.png             # Comparative ROC-AUC Curve Visualization
└── top_10_feature_importances.png       # Horizontal Bar Chart of Top Churn Drivers
```

---

## 🧠 Deep-Dive Model Analysis: Random Forest vs. XGBoost

### 1. Architectural Differences
- **Bagging vs. Boosting:** Random Forest builds multiple independent decision trees in parallel using bootstrap aggregation (bagging) and averages their predictions to reduce variance. XGBoost builds trees sequentially (boosting), where each consecutive tree is trained to minimize the residual errors made by the prior trees.
- **Handling Complex Boundaries:** While Random Forest relies on vote aggregation, XGBoost uses gradient descent optimization on a specified loss function (`logloss`), allowing it to capture finer non-linear feature interactions.

### 2. Generalization & Performance Comparison
- **Cross-Validation vs. Holdout:** Random Forest exhibited slightly higher cross-validation scores during training on SMOTE-generated data, but **XGBoost achieved higher generalization on unseen test data (83.12% vs. 82.45% ROC-AUC)** due to its built-in regularization mechanics (`L1`/`L2`).
- **Robustness to Class Noise:** SMOTE creates synthetic points near class decision boundaries. XGBoost's shrinkage factor (`learning_rate`) helped mitigate over-fitting to noisy SMOTE synthetic data better than Random Forest's unweighted tree voting.

---

## 🚀 Future Enhancements & Next Steps

To upgrade this churn prediction pipeline to enterprise production standard, the following improvements are recommended:
1. **Hyperparameter Optimization:** Integrate `Optuna` or `RandomizedSearchCV` to fine-tune XGBoost hyperparameters (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) and Random Forest parameters (`n_estimators`, `max_features`, `min_samples_leaf`).
2. **Probability Threshold Tuning:** Optimize the classification decision threshold (default `0.5`) to maximize F1-Score or maximize dollar-value business retention metrics based on customer acquisition cost vs. retention cost.
3. **Advanced Model Explainability:** Implement **SHAP (SHapley Additive exPlanations)** values to provide local instance-level explainability for individual high-churn-risk customers.
4. **Pipeline Deployment:** Package the `ColumnTransformer` and trained XGBoost model into a serialized MLflow artifact or FastAPI service for real-time customer churn scoring.

---

## 🔬 Algorithm Deep Dive: Technical Specifics

### 🌲 Random Forest Classifier
* **Paradigm:** Bootstrap Aggregating (Bagging) Ensemble.
* **Mechanism:**
  1. Draws $B$ random bootstrap samples with replacement from the original training dataset.
  2. Trains an unpruned decision tree on each sample. At each split node, only a random subset $m = \sqrt{p}$ features is considered to decorrelate individual trees.
  3. Outputs the final prediction via majority voting (or average predicted probability across all trees):
     $$\hat{P}(Y=1|X) = \frac{1}{B} \sum_{b=1}^{B} T_b(X)$$
* **Strengths:** Highly robust to outliers, non-parametric, inherently parallelizable, low variance without heavy tuning.
* **Weaknesses:** Can become computationally memory-heavy with massive tree counts; less effective when subtle gradient trends dominate the feature space.

### ⚡ XGBoost (eXtreme Gradient Boosting)
* **Paradigm:** Gradient Boosted Decision Trees (GBDT).
* **Mechanism:**
  1. Fits trees sequentially to optimize an objective function combining a convex loss function and a regularization term:
     $$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
  2. Employs a second-order Taylor expansion to rapidly approximate the objective function using first-order gradients ($g_i$) and second-order Hessian values ($h_i$).
  3. Applies regularization $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ to penalize tree complexity and leaf weights ($w$).
* **Strengths:** Industry-leading performance on tabular data, built-in missing value handling, exact/approximate greedy node splitting algorithms, and regularization parameters (`reg_alpha`, `reg_lambda`) that prevent overfitting.
* **Weaknesses:** Requires careful hyperparameter tuning; more sensitive to hyperparameter choices compared to Random Forest.