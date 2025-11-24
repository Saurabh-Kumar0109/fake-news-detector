import re
import os
import joblib
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path: str, text_cols=('title', 'text'), label_col='label') -> pd.DataFrame:
    """Load CSV and combine text columns into a single `text` column. Assumes label_col exists.

    If your CSV already has a `text` column only, set text_cols to ('text',).
    """
    df = pd.read_csv(path)
    # combine available text columns
    texts = []
    for col in text_cols:
        if col in df.columns:
            texts.append(df[col].fillna('').astype(str))
    if not texts:
        raise ValueError(f"None of the text columns {text_cols} found in {path}")
    df['text'] = texts[0]
    if len(texts) > 1:
        for s in texts[1:]:
            df['text'] = df['text'] + ' ' + s
    # normalize label column
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")
    return df[[ 'text', label_col ]].rename(columns={label_col: 'label'})


def clean_text(s: str) -> str:
    s = s or ''
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare_texts(series: pd.Series) -> pd.Series:
    return series.fillna('').astype(str).map(clean_text)


def get_vectorizer(max_features: int = 10000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words='english')


def save_model_and_vectorizer(model, vectorizer, model_path: str, vectorizer_path: str):
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(vectorizer_path) or '.', exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)


def load_model_and_vectorizer(model_path: str, vectorizer_path: str):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer
