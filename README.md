# Data Mining Projects

This repository contains **four independent data mining projects** exploring different techniques in **data preprocessing, classification, clustering, feature engineering, etc**.

Each project is implemented in **Python using Jupyter Notebooks** and focuses on practical applications of **data mining and machine learning methods**.

---

## Repository Structure

```
DataMining-Projects-Codes
│
├── Project1
│   └── DataMining_Prj1.ipynb
│
├── Project2
│   └── DataMining_Prj2.ipynb
|
├── Project3
│   └── DataMining_Prj3.ipynb
|
├── Project4
│   └── DataMining_Prj4_Supervised.ipynb
|   └── DataMining_Prj4_Unsupervised.ipynb

```

---

## Project 1 — Disease Prediction with Regression
A health data mining pipeline that performs preprocessing, dimensionality reduction, and regression modelling on multi-source NHANES survey data to predict disease indicators.

### Overview

This project integrates five NHANES (National Health and Nutrition Examination Survey) data sources, cleans and merges them, and then applies dimensionality reduction techniques followed by regression models to predict two target health outcomes:

- **MCQ220** — Cancer diagnosis indicator
- **MCQ160L** — Liver condition indicator

### Dataset

Five files loaded from Google Drive:

| File | Description |
|------|-------------|
| `demographic.csv` | Participant demographic information |
| `diet.csv` | Dietary intake data |
| `examination.csv` | Physical examination measurements |
| `labs.csv` | Laboratory test results |
| `questionnaire.csv` | Health questionnaire responses |
| `ColumnDefinitions.xlsx` | Definitions for all dataset columns |

> Data is merged on the `SEQN` participant identifier using inner joins.

### Pipeline

#### 1. Pre-processing

- **Drop sparse columns** — columns with more than 50% missing values are removed
- **Replace sentinel values** — coded non-responses (7, 77, 777, 9, 99, 999, etc.) are replaced with `NaN`
- **Remove duplicates** — only the first occurrence of each `SEQN` is kept
- **Impute missing values** — remaining `NaN` values are filled with the column median
- **Encode categorical columns** — string-typed columns in the examination data are label-encoded
- **Outlier detection** — outliers identified using both Z-score (threshold = 3) and IQR methods
- **Outlier handling** — detected outliers are replaced with the column median

#### 2. Merging

The five cleaned dataframes are merged sequentially on `SEQN` (inner join):
```
examination ← labs ← diet ← demographic
```

#### 3. Correlation Analysis & Feature Selection

- A full correlation heatmap is generated for the top 100 most correlated attribute pairs
- Highly correlated feature pairs (|r| > 0.8) are identified and one feature from each pair is selected to reduce redundancy

#### 4. Dimensionality Reduction

Two techniques are applied and compared:

| Method | Details |
|--------|---------|
| **PCA** | StandardScaler + `sklearn` PCA, reduced to 2 components |
| **Gaussian Random Projection** | `sklearn` GaussianRandomProjection, reduced to 2 components |

#### 5. Regression Modelling

Three regression models are trained on each dimensionality-reduced representation for each target variable:

- **Linear Regression**
- **K-Nearest Neighbours Regression**
- **Decision Tree Regression**

Each combination is evaluated using **RMSE** and **MAE**.

### Requirements
```
pandas
numpy
scikit-learn
tensorflow
seaborn
matplotlib
openpyxl
```

---

## Project 2 - Tweets Sentiment Analysis with Classification Models


A multi-class sentiment classification pipeline for tweets, comparing different feature extraction methods and machine learning classifiers.

### Overview

This project builds and evaluates a tweet sentiment classifier that categorises tweets into four classes: **Positive**, **Negative**, **Neutral**, and **Irrelevant**. Multiple combinations of feature representations and classifiers are benchmarked against one another.

### Dataset

Three CSV splits loaded from Google Drive:

| File | Description |
|------|-------------|
| `twitter_training.csv` | Training set |
| `twitter_validation.csv` | Validation set |
| `twitter_test.csv` | Test set |

> Note: Files are encoded in `ISO-8859-1` due to non-UTF-8 characters in tweet content.

### Pipeline

#### 1. Pre-processing

Raw tweet text is cleaned through the following steps:

- **Emoticon replacement** — smile 🙂, laugh 😄, love ❤️, sad 😢, cry 😭, wink 😉 emoticons are replaced with their word equivalents
- **Remove `@mentions`**
- **Remove punctuation, numbers, and special characters**
- **Remove short words** (length ≤ 3)
- **Remove hashtags**
- **Remove URLs**

#### 2. Text Normalisation

Tweets are tokenised and stemmed using NLTK's `PorterStemmer`, then stitched back into strings.

#### 3. Visualisation

Word clouds are generated for the overall dataset and for each sentiment class to understand the most frequent terms per category.

#### 4. Feature Extraction

Four feature representations are explored:

| Feature | Description |
|---------|-------------|
| **Bag-of-Words (BoW)** | `CountVectorizer` — top 1000 features, English stop words removed |
| **TF-IDF** | `TfidfVectorizer` — top 1000 features, English stop words removed |
| **Word2Vec** | Gensim Skip-gram model (200-dim vectors), tweet vectors averaged across tokens |
| **Doc2Vec** | Gensim Doc2Vec model for document-level embeddings |

#### 5. Classification

Three classifiers are trained and evaluated against all four feature types:

- **Logistic Regression**
- **Naive Bayes** (BoW and TF-IDF only)
- **Random Forest** (`n_estimators=400`)
- **XGBoost** (`max_depth=6`, `n_estimators=1000`)

#### 6. Evaluation

Each model combination is evaluated using:

- Accuracy
- F1 Score (macro)
- Precision (macro)
- Recall (macro)
- Confusion Matrix
- Per-class accuracy

### Requirements
```
pandas
numpy
scikit-learn
gensim
nltk
xgboost
wordcloud
matplotlib
seaborn
```


---

## Project 3 - Scientific Articles Clustering Via Covid-19 dataset

An NLP pipeline applied to the CORD-19 (COVID-19 Open Research Dataset) that performs extensive text pre-processing, feature extraction, topic modelling, dimensionality reduction, and unsupervised clustering on scientific paper titles and abstracts.

### Overview

This project processes a large collection of COVID-19 research articles and applies unsupervised learning techniques to discover latent topics and cluster related papers together. Both **TF-IDF** and **Bag-of-Words** feature representations are explored in combination with **LDA topic modelling**, **PCA**, **K-Means**, and **DBSCAN** clustering.

### Dataset

| File | Description |
|------|-------------|
| `all_sources_metadata_2020-03-13.csv` | CORD-19 metadata containing paper titles, abstracts, authors, journals, and more |

Key columns used: `title`, `abstract`

### Pipeline

#### 1. Data Cleaning

- Remove duplicate records based on `title` and `abstract`
- Drop metadata columns not needed for NLP (DOI, authors, journal, licence, etc.)
- Remove rows where either `title` or `abstract` is `NaN`

#### 2. Text Pre-processing

Applied identically to both the `title` and `abstract` columns:

- **Lowercasing**
- **Punctuation removal**
- **Stopword removal** (NLTK English stopwords)
- **Stemming** (Porter Stemmer)
- **Lemmatization** (WordNet Lemmatizer)
- **Emoticon removal** — a comprehensive emoticon dictionary is used to strip text-based emoticons
- **Emoji removal** — Unicode emoji patterns removed via regex
- **URL removal**
- **HTML tag removal**
- **Chat/abbreviation expansion** — common abbreviations (e.g. LOL, BTW, ASAP) expanded to full words
- **Spell checking** — `pyspellchecker` used to correct misspellings

#### 3. Feature Extraction

Two representations are built by concatenating cleaned title and abstract:

| Feature | Details |
|---------|---------|
| **TF-IDF** | `TfidfVectorizer` — full vocabulary |
| **Bag-of-Words** | `CountVectorizer` — top 1000 features, max_df=0.90, min_df=2, English stop words |

#### 4. Topic Modelling

**Latent Dirichlet Allocation (LDA)** with 20 topics is applied to both feature matrices. The top 10 words per topic are printed to help interpret what each topic represents.

#### 5. Dimensionality Reduction

**PCA** (2 components) is applied to the LDA topic outputs to produce 2D representations for visualisation and clustering.

#### 6. Clustering

Two clustering algorithms are evaluated on both feature pipelines:

| Algorithm | Details |
|-----------|---------|
| **K-Means** | k tested from 2–19; elbow method and silhouette scores used to select best k (optimal: 2 clusters) |
| **DBSCAN** | `eps=0.5`, `min_samples=5`; density-based clustering for irregular shapes |

Results are visualised as 2D scatter plots using the PCA-reduced data.

#### 7. Evaluation

Top 10 words per LDA topic are extracted from both TF-IDF and BoW models to interpret the semantic content of each discovered cluster.

### Requirements
```
numpy
pandas
nltk
scikit-learn
gensim
matplotlib
seaborn
yellowbrick
fuzzywuzzy
pyspellchecker
tensorflow
```
---

## Project 4 - Features Ranking for Classifying Datasets

A two-part project tackling an imbalanced binary classification dataset using supervised learning (Project 4 — Supervised) and unsupervised clustering techniques (Project 4 — Unsupervised). Both notebooks operate on the same dataset and address the challenges posed by class imbalance.

### Dataset

| File | Description |
|------|-------------|
| `Dataset.csv` | Tabular dataset with 81 features and a binary `Class Label` column (0 = majority, 1 = minority) |

Features are pre-processed using Z-score normalisation (`StandardScaler`) before model training in both notebooks.

---

### Part A — Supervised Learning (`DataMining_Prj4_Supervised.ipynb`)

#### Overview

Trains and evaluates classification models on the imbalanced dataset, using several strategies to handle the class imbalance: balanced sampling, oversampling (SMOTE), and class weighting.

#### Pipeline

##### 1. Pre-processing
- Feature standardisation with `StandardScaler`
- Train/test split with stratification to preserve class proportions

##### 2. Random Forest

Three variants are compared:

| Variant | Details |
|---------|---------|
| **Standard Random Forest** | `n_estimators=150`; baseline with no imbalance handling |
| **Balanced Random Forest** | `BalancedRandomForestClassifier` from `imbalanced-learn`; internally balances class weights |
| **SMOTE + Standard Random Forest** | Synthetic minority oversampling applied before training; note: runtime exceeds 4 hours |

##### 3. XGBoost

Three variants are compared:

| Variant | Details |
|---------|---------|
| **Standard XGBoost** | `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` |
| **Weighted XGBoost** | `scale_pos_weight` set to the majority/minority class ratio |
| **SMOTE + XGBoost** | SMOTE oversampling applied to training set before XGBoost training |

##### 4. Evaluation Metrics

Each model is evaluated using:
- Accuracy, Precision, Recall, F1-score
- AUC (Area Under the ROC Curve)
- Confusion Matrix (visualised with `ConfusionMatrixDisplay`)
- Repeated Stratified K-Fold cross-validation (10 splits × 3 repeats) for Random Forest variants

##### 5. Feature Importance
- Feature importance extracted and plotted for XGBoost and SMOTE + Random Forest models

#### Requirements
```
scikit-learn
imbalanced-learn
xgboost
matplotlib
pandas
graphtools
```

---

### Part B — Unsupervised Learning (`DataMining_Prj4_Unsupervised.ipynb`)

#### Overview

Applies unsupervised clustering algorithms to the same dataset. The imbalanced class structure is explored both as a whole and split by class label, and feature importance is derived from clustering centroids.

#### Pipeline

##### 1. Exploration & Pre-processing
- Data exploration with `.info()`, `.describe()`, `.head()`
- Feature standardisation with `StandardScaler`
- Correlation heatmap of all 81 features

##### 2. Optimal K Selection (K-Means)

Silhouette scores are computed for k = 2–10 on three subsets:

| Subset | Method |
|--------|--------|
| Full dataset | `MiniBatchKMeans` (large-scale) |
| Class label = 0 (majority) | `MiniBatchKMeans` |
| Class label = 1 (minority) | Standard `KMeans` |

##### 3. Clustering Experiments

**Part 4-1 — Full Dataset K-Means**
- Optimal k found via silhouette score
- Feature importance derived from centroid spread per feature

**Part 4-2 — Per-Class K-Means**
- Separate K-Means models fitted on class 0 and class 1 subsets
- Feature importances computed per subset
- Weighted average of feature importances calculated proportionally to subset size

**Part 4-3 — Advanced Clustering Algorithms**
Applied after PCA reduction to 2 components:

| Algorithm | Details |
|-----------|---------|
| **Chameleon** | Graph-partitioning hierarchical clustering (`k=7`, `knn=20`, `alpha=2.0`); session instability encountered |
| **DBSCAN** | `eps=4.54`, `min_samples=4` on PCA-reduced data; session crashes noted at full scale |
| **BIRCH** | `threshold=0.5`, `branching_factor=81`, `n_clusters=3`; most stable of the three |

##### 4. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Silhouette Score | Cluster cohesion and separation |
| Adjusted Rand Index | Agreement with true class labels |
| V-Measure | Homogeneity and completeness |

##### 5. Feature Importance
- Permutation feature importance computed for BIRCH using V-measure as the scoring metric
- K-Means centroid spread used as a proxy for feature importance

#### Requirements
```
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
metis-python
networkx
graphtools
chameleon_algorithm (cloned from GitHub)
```

> **Note:** The Chameleon algorithm requires METIS to be compiled and installed. See the notebook setup cells for full installation instructions.


---
