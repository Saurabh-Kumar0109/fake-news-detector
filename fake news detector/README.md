# Fake News Detection

This project is a simple Fake News detection pipeline using NLP and classical ML models.

## Dataset

**Option 1: Real Dataset**
- Use the Kaggle "Fake News Dataset" (search "Fake and real news dataset" on Kaggle)
- Download the CSV and place it at `data/train.csv`

**Option 2: Sample Dataset (for testing)**
- Generate synthetic data using the provided script:
```bash
python src\generate_sample_data.py --n-samples 1000 --output data\sample_data.csv
```

## What this project does

- Preprocesses text (cleaning, lowercasing, removing URLs)
- Extracts TF-IDF features
- Trains Logistic Regression and SVM classifiers
- Saves the best model and TF-IDF vectorizer
- Generates evaluation artifacts (confusion matrix, ROC curves, classification reports)
- Provides a CLI to predict on new texts or CSV files
- Includes a Jupyter notebook for interactive exploration

## Requirements

- Python 3.8+

## Setup

```bash
# from the project root
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Basic Usage

### Generate Sample Data

```bash
python src\generate_sample_data.py --n-samples 1000 --output data\sample_data.csv
```

This creates `data\sample_data_train.csv` and `data\sample_data_test.csv`.

### Train a Model

Train on sample data:
```bash
python src\train.py --data-path data\sample_data_train.csv --model-dir models
```

Or train on Kaggle dataset (if downloaded):
```bash
python src\train.py --data-path data\train.csv --model-dir models
```

**Output:**
- `models\best_model.joblib` - Best performing model
- `models\vectorizer.joblib` - TF-IDF vectorizer
- `models\artifacts\` - Evaluation plots and reports
  - Confusion matrices (PNG)
  - ROC curves (PNG)
  - Classification reports (TXT)

### Predict Single Text

```bash
python src\predict.py --model-path models\best_model.joblib --text "The government passed a new law today"
```

Output: `prediction: 1` (0=Fake, 1=Real)

### Predict from CSV

```bash
python src\predict.py --model-path models\best_model.joblib --input-csv data\to_predict.csv --output-csv data\predictions.csv
```

The input CSV should have a `text` column.

### Run Tests

```bash
python -m pytest tests\ -v
```

### Explore with Jupyter Notebook

```bash
jupyter notebook notebooks\exploration.ipynb
```

The notebook includes:
- Data generation and loading
- Exploratory data analysis
- Model training and comparison
- Evaluation visualizations
- Feature importance analysis
- Interactive prediction examples

## Project Structure

```
fake news detector/
├── src/
│   ├── utils.py                  # Data loading, preprocessing, helper functions
│   ├── train.py                  # Training pipeline with evaluation
│   ├── predict.py                # Prediction CLI
│   └── generate_sample_data.py   # Synthetic dataset generator
├── tests/
│   └── test_pipeline.py          # Unit tests
├── notebooks/
│   └── exploration.ipynb         # Interactive exploration notebook
├── models/                       # Saved models (created after training)
│   ├── best_model.joblib
│   ├── vectorizer.joblib
│   └── artifacts/                # Evaluation artifacts
│       ├── *_confusion_matrix.png
│       ├── *_roc_curve.png
│       └── *_classification_report.txt
├── data/                         # Dataset directory
├── requirements.txt
├── README.md
└── .gitignore
```

## Example Results

With the sample synthetic dataset (200 samples):
- **Logistic Regression**: ~100% accuracy
- **SVM**: ~100% accuracy

Note: Perfect accuracy on synthetic data is expected. Real-world datasets will show more realistic performance.

## Notes

- This scaffold is intended for experimentation and learning
- For production use, add proper data validation, logging, hyperparameter search, and model versioning
- The sample data generator creates simple template-based fake/real news for testing purposes
- For better results, use the real Kaggle "Fake News Dataset"

## Next Steps

- Try hyperparameter tuning (GridSearchCV)
- Experiment with deep learning models (LSTM, BERT)
- Add more sophisticated preprocessing (lemmatization, NER)
- Implement cross-validation
- Deploy as a web service
