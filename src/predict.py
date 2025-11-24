import argparse
import os
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_model_and_vectorizer, clean_text


def predict_text(model, vectorizer, text: str):
    t = clean_text(text)
    X = vectorizer.transform([t])
    pred = model.predict(X)
    return int(pred[0])


def predict_csv(model, vectorizer, input_csv: str, output_csv: str, text_col: str = 'text'):
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {input_csv}")
    df['_cleaned_text'] = df[text_col].fillna('').astype(str).map(clean_text)
    X = vectorizer.transform(df['_cleaned_text'])
    df['prediction'] = model.predict(X)
    df.to_csv(output_csv, index=False)
    return output_csv


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--vectorizer-path', type=str, required=False)
    parser.add_argument('--text', type=str, help='Single text to classify')
    parser.add_argument('--input-csv', type=str, help='CSV with a text column to classify')
    parser.add_argument('--output-csv', type=str, help='Output CSV path to write predictions')
    parser.add_argument('--text-col', type=str, default='text', help='Column name for text in input CSV')
    args = parser.parse_args()

    vec_path = args.vectorizer_path or os.path.splitext(args.model_path)[0] + '_vectorizer.joblib'
    model, vectorizer = load_model_and_vectorizer(args.model_path, vec_path)

    if args.text:
        pred = predict_text(model, vectorizer, args.text)
        print('prediction:', pred)
    elif args.input_csv and args.output_csv:
        out = predict_csv(model, vectorizer, args.input_csv, args.output_csv, text_col=args.text_col)
        print('Wrote predictions to', out)
    else:
        print('Provide --text or both --input-csv and --output-csv')
