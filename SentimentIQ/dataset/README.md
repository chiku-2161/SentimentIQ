# SentimentIQ Dataset Documentation

## 1. Overview & Source
This dataset is sourced from the widely cited **Twitter US Airline Sentiment Dataset**, originally collected by **CrowdFlower** (now Appen / Figure Eight) in February 2015 and hosted on Kaggle and open-source machine learning repositories.

- **Primary Source**: CrowdFlower Data for Everyone Library / Kaggle
- **Mirror Repository**: [ruchitgandhi/Twitter-Airline-Sentiment-Analysis](https://github.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis)
- **License**: Public Domain / Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Domain**: Real-world consumer feedback, customer service interactions, and product opinions.

## 2. Dataset Schema
The prepared file `dataset/reviews.csv` contains the following standardized columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `review_text` | String | Raw text content of the customer review / feedback tweet. |
| `sentiment` | String | Labeled target sentiment: `negative`, `neutral`, or `positive`. |
| `airline` | String | Target service or airline company (e.g., Virgin America, United, Delta, Southwest, US Airways, American). |
| `confidence` | Float | Human annotator agreement score / confidence score (range `[0.0, 1.0]`). |

## 3. Class Distribution & Characteristics
After cleaning missing values, removing exact text duplicates, and normalising labels, the dataset comprises **14,427 unique customer reviews**:

| Sentiment Class | Sample Count | Proportion | Imbalance Handling |
| :--- | :--- | :--- | :--- |
| **Negative** | 9,080 | 62.94% | Dominant class representing customer complaints, delays, and service issues. |
| **Neutral** | 3,057 | 21.19% | Inquiries, flight updates, and standard customer interactions. |
| **Positive** | 2,290 | 15.87% | Praise, gratitude, compliments, and successful flight experiences. |
| **Total** | **14,427** | **100.0%** | Handled in training via `class_weight='balanced'` in Scikit-Learn. |

## 4. Reproducibility
To re-download and re-generate `reviews.csv` at any time, run:

```bash
python dataset/download_dataset.py
```

The script automatically fetches the raw data, validates columns, removes corrupted entries, normalizes text encodings, and saves the clean dataset to `dataset/reviews.csv`.
