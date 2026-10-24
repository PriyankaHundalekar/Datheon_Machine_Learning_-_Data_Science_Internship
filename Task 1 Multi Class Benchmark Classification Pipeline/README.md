# Task 1: Multi-Class Benchmark Classification Pipeline (Iris & Wine Datasets)

## Overview
This project is part of the **Datheon Machine Learning & Data Science Internship (ID: DT-2026-2BI5W)**. It implements complete, end-to-end supervised classification machine learning pipelines for classic benchmark datasets (**Iris** and **Wine**) using Python, scikit-learn, Pandas, Seaborn, and Matplotlib.

---

## 📋 Objectives & Requirements Met
- **Dataset Ingestion:** Loaded standard benchmark datasets (`load_iris` / `load_wine`) via Pandas DataFrames.
- **Exploratory Data Analysis:** Generated and saved exploratory pairplots using Seaborn (`exploratory_pairplot.png`).
- **Feature Normalization:** Applied `StandardScaler` to standardize continuous feature ranges.
- **Train/Test Splitting:** Split datasets cleanly using an 80/20 stratified split.
- **Model Training:** Trained two distinct classifiers on both pipelines:
  1. `LogisticRegression`
  2. `DecisionTreeClassifier`
- **Performance Benchmarking:** Evaluated models using Accuracy, Precision, Recall, F1-Score, and detailed classification reports.
- **Visualizations:** Generated and saved confusion matrix heatmaps (`confusion_matrices.png`) and 2D decision boundary plots (`decision_boundaries.png`).
- **Deliverable Format:** Packaged into interactive Jupyter Notebooks (`iris_classification_pipeline.ipynb` and `wine_classification_pipeline.ipynb`).

---

## 📊 Performance Summary

### 1. Wine Dataset Performance
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **97.22%** | **97.41%** | **97.22%** | **97.20%** |
| **Decision Tree Classifier** | **94.44%** | **95.14%** | **94.44%** | **94.50%** |

### 2. Iris Dataset Performance
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **97.33% - 100%** | **1.00** | **1.00** | **1.00** |
| **Decision Tree Classifier** | **93.33% - 97.00%** | **0.97** | **0.97** | **0.97** |

### **Key Insights:**
- **Iris Dataset:** Exhibited near-perfect linear separability (especially for *Iris-setosa*), allowing Logistic Regression to achieve flawless test separation. Petal length and width served as the strongest discriminators.
- **Wine Dataset:** Logistic Regression outperformed the Decision Tree due to the smooth, continuous linear relationships of chemical measurements. The Decision Tree handled multi-class orthogonal splits well, but was slightly more prone to minor variance on smaller sample sizes.

---

## 🗂️ Project Repository Structure
```text
Task 1 Multi Class Benchmark Classification Pipeline/
│
├── iris_classification_pipeline.ipynb # IRIS DATASET Jupyter Notebook Deliverable
├── wine_classification_pipeline.ipynb # WINE DATASET Jupyter Notebook Deliverable
├── README.md                          # Project Documentation & Summary
├── exploratory_pairplot.png           # Seaborn Feature Distribution Pairplot
├── confusion_matrices.png             # Side-by-side Confusion Matrix Heatmaps
└── decision_boundaries.png            # 2D Decision Boundary Visualizations
```

## 🧠 Deep-Dive Model Analysis: Logistic Regression vs. Decision Tree
1.Why Did Logistic Regression Excel?

- Linear Separability: Both the Iris and Wine datasets feature attributes with clean linear progressions. 
- Logistic Regression fits smooth linear hyperplanes through feature space, perfectly aligning with their natural distribution.

- Low Variance: On compact datasets, Logistic Regression acts as a robust model with low variance, successfully resisting overfitting compared to unconstrained trees.

2.Decision Tree Behavior & Limitations

- Axis-Aligned Splits: Decision trees partition space using orthogonal splits (e.g., Feature A > Threshold), creating step-like decision boundaries. While highly interpretable, they require sufficient depth or ensemble structures to smoothly approximate diagonal linear trends.

- Overfitting Tendency: Without pruning or constraints (max_depth), trees can isolate noise in smaller training partitions, reflected in slightly lower generalization scores compared to linear models.

## 🚀 Future Enhancements & Next Steps
To take these classification pipelines to an enterprise production standard, future iterations can incorporate:

- Hyperparameter Tuning: Implement GridSearchCV or RandomizedSearchCV to optimize regularization parameters (C for Logistic Regression) and tree pruning parameters (max_depth, min_samples_split for Decision Trees).

- Advanced Ensembles: Introduce ensemble methods like Random Forests or Gradient Boosting (XGBoost / LightGBM) to capture non-linear feature interactions and boost overall accuracy.

- Cross-Validation: Replace the static 80/20 split with Stratified K-Fold Cross-Validation to ensure robust performance across multiple folds of the dataset.

- Model Explainability: Integrate SHAP (SHapley Additive exPlanations) or LIME values to visualize feature importances and interpret individual prediction decisions.