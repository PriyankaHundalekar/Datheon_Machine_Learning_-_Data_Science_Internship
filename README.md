# Datheon Machine Learning & Data Science Internship Portfolio

Welcome to my repository for the **Datheon Machine Learning & Data Science Internship (ID: DT-2026-2BI5W)**. This repository houses production-grade, reproducible Jupyter Notebook implementations covering supervised classification, supervised regression, ensemble methods for customer churn, and natural language processing (NLP) sentiment analysis.

---

## 📂 Repository Architecture & Task Directory

```text
Datheon_Machine_Learning_&_Data_Science_Internship/
│
├── Task_1_Multi_Class_Classification/
│   ├── iris_classification_pipeline.ipynb # Iris benchmark classification & decision boundaries
│   ├── wine_classification_pipeline.ipynb # Wine chemical benchmark classification
│   └── README.md                         # Detailed Task 1 documentation
│
├── Task_2_Supervised_Regression_Pipeline/
│   ├── housing_regression_pipeline.ipynb  # California housing regression & parity plots
│   └── README.md                         # Detailed Task 2 documentation
│
├── Task_3_Telecom_Churn_Prediction/
│   ├── churn_prediction_pipeline.ipynb   # Ensemble churn prediction (Random Forest & XGBoost)
│   └── README.md                         # Detailed Task 3 documentation
│
├── Task_4_NLP_Sentiment_Analysis/
│   ├── nlp_sentiment_pipeline.ipynb      # NLP sentiment classification with NLTK & TF-IDF
│   └── README.md                         # Detailed Task 4 documentation
│
└── README.md                             # Root portfolio overview (This file)

```

---

## 📋 Summary of Completed Internship Projects
🔹Task 1: Multi-Class Benchmark Classification PipelineDatasets: 

Iris & Wine Benchmark Datasets (scikit-learn).Pipeline Highlights: Applied feature standardization (StandardScaler), 80/20 stratified train-test splits, and trained both Logistic Regression and Decision Tree Classifiers.Key Results: Achieved high benchmark accuracy (>94% to 97%+ across datasets) with visualized confusion matrices and 2D decision boundaries.

🔹 Task 2: Supervised Regression PipelineDataset: 

California Housing Dataset (scikit-learn).Pipeline Highlights: Built an end-to-end regression pipeline applying feature scaling (StandardScaler), train-test splitting, and training both Linear Regression and Random Forest Regressor models.Key Results: Evaluated models using RMSE, MAE, and $R^2$ Score, demonstrating the superior performance of non-linear ensembles on complex property valuation metrics.

🔹 Task 3: Telecom Customer Churn Prediction & Ensemble ModelingDataset: 

IBM Telco Customer Churn Subscription Records.Pipeline Highlights: Handled missing continuous values, encoded categorical features using scikit-learn's ColumnTransformer and OneHotEncoder, resolved class imbalance using SMOTE (Synthetic Minority Over-sampling Technique), and cross-validated ensemble models.Key Results: Trained Random Forest and XGBoost classifiers, successfully crossing the >80% ROC-AUC benchmark threshold and extracting the top 10 most influential feature drivers of churn.

🔹 Task 4: Natural Language Processing (NLP) Sentiment AnalysisDataset: 

IMDB Movie Reviews Dataset.Pipeline Highlights: Built a custom text-cleaning pipeline utilizing Python regular expressions (RegEx), NLTK stopword removal, and WordNetLemmatizer. Transformed text using TfidfVectorizer(max_features=5000).Key Results: Benchmarked Multinomial Naive Bayes and Logistic Regression, and implemented a live interactive predict_sentiment() helper function to evaluate custom user input phrases with confidence scoring.

## 🛠️ Tech Stack & Dependencies

All notebooks are fully self-contained with inline package installation blocks and run on a robust Python data science stack:Core Data Science: Pandas, NumPy, Scikit-LearnVisualization: Matplotlib, SeabornImbalanced Learning: Imbalanced-learn (SMOTE)Gradient Boosting: XGBoostNatural Language Processing: NLTK (stopwords, WordNetLemmatizer)Alternatively, you can install all required dependencies at once using:Bashpip install -r requirements.txt
