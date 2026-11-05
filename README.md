# SentimentIQ – AI-Powered Sentiment Analysis and Customer Feedback Intelligence Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1.0-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/Tests-26%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org/)

---

## 1. Project Banner Placeholder

```
========================================================================================
   ____             _   _                      _   ___ ____  
  / ___|  ___ _ __ | |_(_)_ __ ___   ___ _ __ | |_|_ _|  _ \ 
  \___ \ / _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __|| || | | |
   ___) |  __/ | | | |_| | | | | | |  __/ | | | |_ | || |_| |
  |____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__|___|___\_\
                                                             
  AI-Powered Customer Feedback Intelligence & Real-Time Sentiment Classification System
========================================================================================
```

---

## 2. Problem Statement

Modern enterprise customer support, product development, and operations teams receive thousands of unsolicited feedback items daily across customer review boards, social media channels, support tickets, and satisfaction surveys. 

Manually sorting through these feedback volumes is slow, costly, and error-prone. Critical customer frustrations (cancellations, delays, defects) risk being lost in the noise, while positive brand advocacy cannot be identified and leveraged in real time. 

Organizations require an automated, low-latency, interpretable, and reproducible machine learning solution capable of categorizing text feedback into actionable sentiment classes (**Positive**, **Negative**, and **Neutral**), surfacing class probabilities, processing bulk batches, and extracting descriptive feedback patterns.

---

## 3. Project Objectives

1. **Production-Grade ML Pipeline**: Implement an end-to-end sentiment classification workflow using TF-IDF feature extraction and class-balanced Logistic Regression.
2. **Prevent Data Leakage**: Enforce strict train-test segregation (`80:20` stratified split) where feature vectorization parameters are learned solely from training data.
3. **Negation-Preserving NLP**: Develop text normalization routines that preserve essential negation terms (`not`, `never`, `cannot`, `no`) and contraction expansions.
4. **Interactive Intelligence Dashboard**: Provide a responsive 5-page Streamlit dashboard for single-text inference, bulk CSV batch analytics, empirical model performance evaluation, and linguistic intelligence.
5. **Robust Validation & Security**: Enforce file-size ceilings (`10 MB`), CSV structure validation, empty-input fallbacks, and zero unvalidated code execution.
6. **Reproducible & GitHub Ready**: Provide a fully tested codebase (26 Pytest tests), standardized environment setup, and honest empirical benchmarks.

---

## 4. Features

- **Single Text Inference**: Instant classification with confidence gauges, per-class probability bars, normalized token visualization, and quick preset scenarios.
- **Bulk CSV Batch Processing**: Upload customer review spreadsheets, auto-detect text columns, run high-throughput vectorized inference, filter by sentiment class, and export enriched CSVs.
- **Interactive Visualizations**: Dynamic Plotly donut charts, sentiment volume bar charts, confidence distribution histograms, and interactive data tables.
- **Auditable Model Performance**: Direct visualization of confusion matrices, precision, recall, and F1-scores derived exclusively from held-out test records.
- **Feedback Pattern Discovery**: Extract top unigrams and bigrams across positive and negative feedback categories with exportable JSON intelligence summaries.

---

## 5. Technology Stack

- **Core Runtime**: Python 3.10+
- **Data Manipulation**: Pandas 2.3+, NumPy 1.26+
- **Machine Learning & NLP**: Scikit-Learn 1.7+, Joblib 1.5+
- **Vectorization**: TF-IDF Vectorizer (Unigrams & Bigrams, sublinear TF)
- **Web UI & Dashboard**: Streamlit 1.57+
- **Interactive Visualizations**: Plotly 7.1+
- **Automated Testing**: Pytest 9.1+
- **HTTP / Dataset Sourcing**: Requests 2.32+

---

## 6. System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A[Raw Customer Feedback CSV] --> B[dataset/download_dataset.py]
        B --> C[dataset/reviews.csv<br/>14,427 Standardized Records]
    end

    subgraph Preprocessing & Feature Extraction
        C --> D[src/preprocessing.py<br/>HTML/URL Removal, Mention Stripping, Contraction Expansion]
        D --> E[Stratified Train-Test Split<br/>80% Train: 11,541 / 20% Test: 2,886]
        E --> F[TF-IDF Vectorizer<br/>10,000 Features, N-gram: 1-2]
    end

    subgraph Model & Training
        F --> G[Logistic Regression Classifier<br/>class_weight='balanced', max_iter=1000]
        G --> H[Saved Artifacts<br/>models/sentiment_model.joblib<br/>models/tfidf_vectorizer.joblib]
        G --> I[src/evaluation.py<br/>models/evaluation_metrics.json]
    end

    subgraph Inference & Analytics
        H --> J[src/predict.py<br/>SentimentPredictor Singleton Engine]
        I --> K[app.py Streamlit Dashboard]
        J --> K
        L[src/analytics.py<br/>N-Gram Frequency & Linguistic Intelligence] --> K
        M[src/utils.py<br/>File Size Limits & Schema Validation] --> K
    end
```

---

## 7. Complete Folder Structure

```
SentimentIQ/
├── app.py                         # Interactive Streamlit Multi-Page Dashboard
├── train_model.py                 # CLI script to execute model training & evaluation
├── requirements.txt               # Locked and verified dependencies
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules for cache, venv, and temporary outputs
├── LICENSE                        # MIT Open-Source License
├── config/
│   └── settings.py                # Hyperparameters, directory paths, and color themes
├── src/
│   ├── __init__.py                # Package initializer
│   ├── preprocessing.py          # Negation-preserving text normalization routines
│   ├── train.py                  # Stratified training pipeline and artifact serialization
│   ├── predict.py                # Thread-safe SentimentPredictor inference engine
│   ├── evaluation.py             # Accuracy, F1, confusion matrix, and metric persistence
│   ├── analytics.py              # N-gram frequency discovery and feedback statistics
│   └── utils.py                  # CSV loading, file size bounds, and column detection
├── dataset/
│   ├── reviews.csv               # Cleaned 3-class sentiment dataset (14,427 rows)
│   ├── download_dataset.py       # Reproducible script to fetch and prepare dataset
│   └── README.md                 # Dataset documentation, schema, and licensing
├── models/
│   ├── sentiment_model.joblib    # Trained Logistic Regression model artifact
│   ├── tfidf_vectorizer.joblib   # Fitted TF-IDF vectorizer artifact
│   └── evaluation_metrics.json   # Actual empirical held-out test evaluation results
├── tests/
│   ├── __init__.py               # Test package initializer
│   ├── test_preprocessing.py     # Unit tests for normalization and negations
│   ├── test_prediction.py        # Unit tests for inference, confidence, and batches
│   └── test_data_validation.py   # Unit tests for CSV safety, file size, and schema
├── screenshots/
│   └── .gitkeep                  # Screenshots directory placeholder
└── outputs/
    └── .gitkeep                  # Directory for generated CSV batch exports
```

---

## 8. Dataset Source & Licensing Information

- **Dataset**: Twitter US Airline Sentiment Dataset (Customer Service & Feedback Domain)
- **Original Source**: CrowdFlower / Figure Eight (Appen Data for Everyone Library)
- **Hosted Mirror**: [ruchitgandhi/Twitter-Airline-Sentiment-Analysis](https://github.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain
- **Total Valid Records**: **14,427**

### Dataset Schema (`dataset/reviews.csv`)
| Column Name | Type | Description |
| :--- | :--- | :--- |
| `review_text` | String | Customer review text or feedback tweet |
| `sentiment` | String | Target class label: `negative`, `neutral`, `positive` |
| `airline` | String | Target company / carrier entity |
| `confidence` | Float | Human annotator agreement score (`0.0` - `1.0`) |

### Class Distribution
| Sentiment Class | Sample Count | Proportion | Imbalance Treatment |
| :--- | :--- | :--- | :--- |
| **Negative** | 9,080 | 62.94% | Handled via inverse class frequency weighting |
| **Neutral** | 3,057 | 21.19% | Calibrated multi-class probability estimation |
| **Positive** | 2,290 | 15.87% | Sublinear TF feature scaling to balance impact |
| **Total** | **14,427** | **100.0%** | Stratified split across all training/test folds |

---

## 9. Machine Learning Workflow

1. **Data Ingestion**: Standardize schema, discard null rows, remove duplicates.
2. **Text Preprocessing**:
   - Strip HTML tags and entities.
   - Clean web URLs (`http`, `https`, `www`).
   - Remove user `@mentions` to prevent entity bias.
   - Expand contractions (e.g., `can't` $\rightarrow$ `cannot`, `won't` $\rightarrow$ `will not`).
   - Retain negation operators (`not`, `never`, `no`).
   - Normalize punctuation and whitespace.
3. **Stratified Split**: Partition records into `80%` Training (11,541 rows) and `20%` Test (2,886 rows) using stratified sampling.
4. **Leakage-Free Vectorization**: Fit TF-IDF solely on training records with `max_features=10,000` and `ngram_range=(1, 2)`. Transform test set.
5. **Class-Balanced Training**: Fit Logistic Regression using the `lbfgs` solver and `class_weight='balanced'` to prevent majority class dominance.
6. **Empirical Evaluation**: Evaluate predictions on the 2,886 held-out test records. Persist model, vectorizer, and metrics.

---

## 10. Actual Measured Model Evaluation Metrics

> [!NOTE]
> All metrics displayed below were empirically measured on the held-out test set (`N = 2,886`). None of the figures are fabricated or estimated.

### Overall Performance Summary
| Metric | Measured Score | Evaluation Notes |
| :--- | :--- | :--- |
| **Overall Accuracy** | **79.07%** | Proportion of correctly classified test feedback items |
| **Macro Precision** | **73.58%** | Unweighted average precision across all 3 classes |
| **Macro Recall** | **76.25%** | Unweighted average recall across all 3 classes |
| **Macro F1-Score** | **74.71%** | Harmonic mean of macro precision and macro recall |
| **Weighted F1-Score** | **79.54%** | F1-score weighted by support of each class |
| **Total Test Samples**| **2,886** | Strict 20% stratified held-out evaluation slice |

### Per-Class Detailed Breakdown
| Sentiment Class | Precision | Recall | F1-Score | Test Support ($N$) |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | **0.896** | **0.830** | **0.862** | 1,816 records |
| **Neutral** | **0.594** | **0.698** | **0.642** | 612 records |
| **Positive** | **0.718** | **0.760** | **0.738** | 458 records |

### Confusion Matrix (Test Set $N = 2,886$)
| True \ Predicted | Predicted Negative | Predicted Neutral | Predicted Positive |
| :--- | :---: | :---: | :---: |
| **Actual Negative** | **1,507** | 234 | 75 |
| **Actual Neutral** | 123 | **427** | 62 |
| **Actual Positive** | 52 | 58 | **348** |

---

## 11. Installation & Setup Instructions

### Prerequisites
- Python 3.10 or higher installed.
- Git installed.

### Virtual Environment Setup (Windows PowerShell)

```powershell
# 1. Clone the repository or navigate to project directory
cd "c:\Users\PABITRA KUMAR BARIK\OneDrive\SentimentIQ"

# 2. Create a virtual environment named .venv
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

*(For Linux/macOS users: `source .venv/bin/activate`)*

### Dependency Installation

```powershell
# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. Model Training & Dashboard Execution

### Step 1: Prepare the Dataset
Fetch, validate, and standardize the dataset:

```powershell
python dataset/download_dataset.py
```

### Step 2: Train the Machine Learning Pipeline
Execute model training, feature extraction, and held-out evaluation:

```powershell
python train_model.py
```

*Optional command flags:*
```powershell
python train_model.py --dataset dataset/reviews.csv --test-size 0.20 --random-state 42
```

### Step 3: Launch the Interactive Streamlit Dashboard

```powershell
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 13. Dashboard Usage Guide

The Streamlit dashboard is structured into 5 navigation views:

1. **📊 Executive Overview**:
   - Instant KPI tiles: Total Reviews, Positive %, Neutral %, and Negative %.
   - Interactive Plotly donut chart and volume distribution bar chart.
   - Feedback explorer with keyword search and sentiment class filters.
2. **🔍 Single Text Analysis**:
   - Interactive text input area with 3 pre-configured scenario presets.
   - Color-coded prediction badge, confidence score progress meter, and normalized text breakdown.
   - Horizontal probability distribution chart across Positive, Neutral, and Negative.
3. **📁 Bulk CSV Batch Analysis**:
   - Secure file uploader supporting datasets up to `10 MB`.
   - Downloadable ready-to-test sample CSV template (`sentiment_sample_reviews.csv`).
   - Column selector with automatic text column detection.
   - Progress-tracked batch inference and instant export of enriched CSV results.
4. **🎯 Model Performance**:
   - Full evaluation report with Confusion Matrix heatmap and per-class metrics.
   - Complete architectural parameters and split methodology.
5. **💡 Feedback Intelligence**:
   - Top descriptive n-grams discovered across positive and negative feedback categories.
   - Comparative review length statistics and downloadable JSON intelligence report.

---

## 14. Example Inputs & Outputs

### Example 1: Enthusiastic Positive Feedback
- **Input Text**: `"Outstanding flight! The cabin crew was polite, attentiveness was top notch, and we landed 20 minutes early."`
- **Normalized Text**: `outstanding flight! the cabin crew was polite attentiveness was top notch and we landed 20 minutes early.`
- **Predicted Sentiment**: `positive`
- **Confidence**: `94.6%`
- **Probabilities**: `Positive: 0.946 | Neutral: 0.012 | Negative: 0.043`

### Example 2: Negative Delay Complaint
- **Input Text**: `"Worst travel experience of my life. Tarmac delay of 4 hours, lost my bags, and customer care was completely rude!"`
- **Normalized Text**: `worst travel experience of my life. tarmac delay of 4 hours lost my bags and customer care was completely rude!`
- **Predicted Sentiment**: `negative`
- **Confidence**: `93.6%`
- **Probabilities**: `Negative: 0.936 | Neutral: 0.025 | Positive: 0.039`

### Example 3: Neutral Operational Inquiry
- **Input Text**: `"Could you tell me what terminal flight 782 departs from tomorrow morning?"`
- **Normalized Text**: `could you tell me what terminal flight 782 departs from tomorrow morning?`
- **Predicted Sentiment**: `neutral`
- **Confidence**: `62.5%`
- **Probabilities**: `Neutral: 0.625 | Negative: 0.298 | Positive: 0.077`

---

## 15. Automated Testing Suite

The project includes 26 comprehensive unit and integration tests covering preprocessing, model prediction, and data validation.

To run the complete Pytest suite:

```powershell
pytest tests -v
```

### Test Coverage Summary
- `tests/test_preprocessing.py` (12 tests):
  - Lowercase normalization, HTML tag stripping, URL removal, mention stripping.
  - Negation retention (`not`, `never`, `cannot`, `no`), contraction expansion (`won't` $\rightarrow$ `will not`).
  - Redundant whitespace collapsing, punctuation normalization.
  - Edge cases: empty text, whitespace-only, `None`, `np.nan`, non-string integers, Series vectorization.
- `tests/test_prediction.py` (8 tests):
  - Model artifact loading and class alignment.
  - Single prediction dictionary structure and confidence score bounds ($[0.0, 1.0]$).
  - Class probabilities summing to $\approx 1.0$.
  - Empty text fallback to graceful neutral prediction.
  - Positive and negative inference accuracy.
  - Batch DataFrame inference shape, types, and column presence.
- `tests/test_data_validation.py` (6 tests):
  - File size threshold validation ($> 10\text{ MB}$ rejected).
  - Valid CSV parsing from streams and buffers.
  - Empty CSV detection and descriptive error raising.
  - Missing column detection.
  - Review text column auto-detection and heuristic fallback.

---

## 16. Screenshots Placeholders

Place application screenshots inside the `screenshots/` directory using these file paths:

- **Overview Dashboard**: `screenshots/01_overview_dashboard.png`
- **Single Text Inference**: `screenshots/02_single_text_analysis.png`
- **Bulk CSV Analysis**: `screenshots/03_bulk_csv_analysis.png`
- **Model Evaluation**: `screenshots/04_model_performance.png`
- **Feedback Intelligence**: `screenshots/05_feedback_intelligence.png`

---

## 17. Model Interpretability & Limitations

> [!WARNING]
> **Probabilistic Model Notice**: Sentiment predictions generated by SentimentIQ are statistical estimates produced by a TF-IDF Logistic Regression model trained on real-world customer feedback. Predictions represent probabilistic indicators and should **not** be treated as infallible human judgments or legally binding declarations.

### Known Limitations:
1. **Subtle Sarcasm & Irony**: Sarcastic remarks (e.g., *"Oh fantastic, another 4-hour delay, just what I needed!"*) can occasionally trigger positive scores due to positive vocabulary (`fantastic`) despite negative underlying intent.
2. **Context-Free Neutrality**: Short phrases containing flight numbers or acronyms may exhibit lower confidence thresholds due to sparse n-gram overlap.
3. **Domain Vocabulary**: The reference model was trained on airline and customer service feedback. While it generalizes well to product and hospitality feedback, domain-specific terminology (e.g., biomedical or financial jargon) may require domain-adapted fine-tuning.

---

## 18. Future Improvements

- [ ] Incorporate transformer-based embeddings (e.g., DistilRoBERTa) for nuanced sarcasm and context detection.
- [ ] Add Aspect-Based Sentiment Analysis (ABSA) to tag sentiments to specific feedback entities (e.g., Staff vs. Luggage vs. Pricing).
- [ ] Implement multilingual translation and sentiment classification support.
- [ ] Add RESTful API endpoints using FastAPI for programmatic microservice integration.

---

## 19. Contribution Guidelines

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Commit your changes: `git commit -m "Add amazing feature"`.
4. Run the test suite: `pytest tests -v` (ensure all 26 tests pass).
5. Push to the branch: `git push origin feature/amazing-feature`.
6. Open a Pull Request.

---

## 20. License

This project is licensed under the **MIT License**. See the [LICENSE](file:///c:/Users/PABITRA%20KUMAR%20BARIK/OneDrive/SentimentIQ/LICENSE) file for complete details.

---

## 21. Author

**SentimentIQ Development Team**
- GitHub Profile: [@chiku-2161](https://github.com/chiku-2161)
- Project Repository: [https://github.com/chiku-2161/SentimentIQ](https://github.com/chiku-2161/SentimentIQ/tree/main)
