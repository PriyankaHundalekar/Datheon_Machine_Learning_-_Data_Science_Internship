# Task 4: Natural Language Processing (NLP) Sentiment Analysis Pipeline

## Overview

This project is part of the **Datheon Machine Learning & Data Science Internship (ID: DT-2026-2BI5W)**. It delivers an end-to-end Natural Language Processing (NLP) classification pipeline designed to categorize customer and movie review records into **Positive** or **Negative** sentiment.

The pipeline incorporates robust text preprocessing techniques (lowercasing, HTML stripping, punctuation removal, stopword removal, and lemmatization via NLTK), feature engineering through **TF-IDF vectorization**, model benchmarking using **Multinomial Naive Bayes** and **Logistic Regression**, and an interactive helper function for real-time live phrase testing.

## 📋 Objectives & Requirements Met

* **Dataset Ingestion:** Ingested the IMDB Sentiment Reviews benchmark dataset featuring 50,000 balanced binary sentiment records.

* **Text Preprocessing & Cleaning:** Constructed a custom cleaning pipeline utilizing Python `re` (RegEx) and `nltk`:

  * Converted text to lowercase.

  * Stripped HTML tags (e.g., `<br />`) and non-alphabet characters/punctuation.

  * Removed standard English stopwords.

  * Applied word lemmatization via NLTK's `WordNetLemmatizer`.

* **Feature Extraction:** Transformed unstructured text into numerical feature matrices using `TfidfVectorizer` configured with `max_features=5000` and unigram/bigram token ranges (`ngram_range=(1, 2)`).

* **Model Training & Benchmarking:** Trained and benchmarked two distinct classification algorithms:

  1. `MultinomialNB` (Multinomial Naive Bayes)

  2. `LogisticRegression`

* **Performance Evaluation:** Evaluated models using out-of-sample Accuracy, ROC-AUC, precision, recall, weighted F1-score, and confusion matrix heatmaps.

* **Live Inference Helper:** Implemented `predict_sentiment()` to preprocess, vectorize, and evaluate custom live user phrases on-the-fly with confidence percentages.

* **Deliverable Format:** Packaged into an interactive Jupyter Notebook (`nlp_sentiment_analysis.ipynb`) and comprehensive project documentation.

## 📊 Performance Summary

| Model | Accuracy | ROC-AUC Score | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | **88.62%** | **0.9542** | **0.8864** | **0.8862** | **0.8862** | 
| **Multinomial Naive Bayes** | **85.10%** | **0.9275** | **0.8518** | **0.8510** | **0.8509** | 

*(Note: Evaluated on an 80/20 stratified train-test split containing 10,000 unseen test reviews).*

### **Key Insights:**

* **Logistic Regression Superiority:** Logistic Regression outperformed Multinomial Naive Bayes by **\~3.5% in accuracy**, benefiting from its ability to learn continuous feature weights across collinear TF-IDF n-grams.

* **Impact of Lemmatization & TF-IDF:** Filtering out noisy stopwords and reducing inflected words to their dictionary lemma (e.g., "running", "ran" $\rightarrow$ "run") significantly reduced matrix sparsity, optimizing memory and vocabulary representation.

* **Handling Negations & Context:** Bigram inclusion (`ngram_range=(1, 2)`) captured critical contextual sentiment modifiers (e.g., "not good", "highly recommended").

## 🗂️ Project Repository Structure

```
Task 4 NLP Sentiment Analysis/
│
├── nlp_sentiment_analysis.ipynb         # Main Jupyter Notebook Pipeline
├── README.md                            # Comprehensive Project Documentation
└── nlp_confusion_matrices.png           # Side-by-side Confusion Matrix Heatmaps

```

## 🧪 Interactive Inference Function

The pipeline includes an interactive prediction helper function `predict_sentiment(text, model_name)` for evaluating live phrases:

```
# Example Usage
predict_sentiment("This movie was an absolute masterpiece! Great acting and stunning visuals.")

```

**Sample Output:**

```
Input Text:          "This movie was an absolute masterpiece! Great acting and stunning visuals."
Predicted Sentiment: Positive 😊
Confidence:          96.48%

```

## 🔬 Algorithm Deep Dive: Technical Specifics

### 📐 1. Multinomial Naive Bayes (`MultinomialNB`)

* **Paradigm:** Probabilistic Generative Classifier based on Bayes' Theorem.

* **Mechanism:**
  Computes the posterior probability of a review belonging to class $y \in \{0, 1\}$ given feature vector $\mathbf{x} = (x_1, x_2, \dots, x_n)$ representing TF-IDF term frequencies:
  

  $$
  P(y \mid \mathbf{x}) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)^{x_i}}{P(\mathbf{x})}
  $$

  
  Applies **Laplace Smoothing** ($\alpha = 1.0$) to avoid zero-probability issues for unseen words in a class:
  

  $$
  \hat{P}(x_i \mid y) = \frac{N_{yi} + \alpha}{N_y + \alpha n}
  $$

  
  *(Where* $N_{yi}$ *is the sum of TF-IDF weights for feature* $i$ *in class* $y$*, and* $N_y$ *is the total term count for class* $y$*).*

* **Strengths:** Exceptionally fast training and inference speeds, low computational overhead, performs remarkably well on high-dimensional sparse text data.

* **Weaknesses:** Assumes conditional independence between features given the class label (the "naive" assumption), which ignores word order and strong feature dependencies.

### 📈 2. Logistic Regression (`LogisticRegression`)

* **Paradigm:** Discriminative Linear Model with Logit Link Function.

* **Mechanism:**
  Models the log-odds of positive sentiment as a linear combination of input TF-IDF feature weights:
  

  $$
  z = w_0 + \sum_{i=1}^{n} w_i x_i
  $$

  
  Passes the linear log-odds through the **Sigmoid (logistic) activation function** to map real-valued scores into bounded probability space $[0, 1]$:
  

  $$
  P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}
  $$

  
  Optimizes model weights by minimizing the binary cross-entropy (log-loss) function with $L_2$ regularization penalty:
  

  $$
  \mathcal{L}(w) = -\frac{1}{N} \sum_{j=1}^{N} \left[ y_j \log(\hat{y}_j) + (1 - y_j) \log(1 - \hat{y}_j) \right] + \frac{1}{2C} \Vert{}w\Vert{}_2^2
  $$

* **Strengths:** Captures linear feature interactions effectively, provides smooth probabilistically calibrated outputs, robust against overfitting on high-dimensional sparse text due to regularization ($C$).

* **Weaknesses:** Cannot naturally capture non-linear relationships without explicit feature engineering (e.g., higher-order n-grams or dense embeddings).

## 🚀 Future Enhancements & Next Steps

To upgrade this NLP sentiment analysis pipeline to enterprise production standards:

1. **Advanced Embeddings:** Replace static TF-IDF representations with dense contextual word embeddings like **Word2Vec**, **GloVe**, or transformer-based embeddings (**BERT / RoBERTa**).

2. **Deep Learning Architecture:** Introduce Recurrent Neural Networks (**LSTM / GRU**) or fine-tuned Hugging Face transformer models to better capture long-range word dependencies and subtle sarcasm.

3. **Hyperparameter Tuning:** Fine-tune `C` regularization in Logistic Regression and `alpha` smoothing parameters in Naive Bayes using `GridSearchCV`.

4. **API Deployment:** Wrap the preprocessor and trained pipeline into a lightweight REST API endpoint using **FastAPI** or **Flask** for continuous real-time sentiment scoring.