import argparse
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data, prepare_texts, get_vectorizer, save_model_and_vectorizer


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_scores, model_name: str, save_path: str):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def train(data_path: str, model_dir: str, test_size: float = 0.2, random_state: int = 42):
    df = load_data(data_path)
    df['text'] = prepare_texts(df['text'])
    # Map labels to numeric if needed
    labels = df['label']
    if labels.dtype == object:
        labels = labels.map(lambda s: 1 if str(s).lower() in ('real', 'true', '1') else 0)
    X = df['text']
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    vectorizer = get_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'logreg': LogisticRegression(max_iter=1000),
        'svm': LinearSVC(max_iter=10000)
    }

    results = {}
    artifacts_dir = os.path.join(model_dir, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {acc:.4f}")
        
        # Classification report
        report = classification_report(y_test, preds, target_names=['Fake', 'Real'])
        print(report)
        
        # Save classification report
        report_path = os.path.join(artifacts_dir, f'{name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(report)
        print(f"Saved classification report to {report_path}")
        
        # Plot confusion matrix
        cm_path = os.path.join(artifacts_dir, f'{name}_confusion_matrix.png')
        plot_confusion_matrix(y_test, preds, name, cm_path)
        
        # Plot ROC curve (need decision function or probabilities)
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test_tfidf)
        elif hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test_tfidf)[:, 1]
        else:
            y_scores = preds  # fallback
        
        roc_path = os.path.join(artifacts_dir, f'{name}_roc_curve.png')
        try:
            plot_roc_curve(y_test, y_scores, name, roc_path)
            roc_auc = roc_auc_score(y_test, y_scores)
            print(f"{name} ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not plot ROC curve for {name}: {e}")
        
        results[name] = (model, acc)

    # choose best by accuracy
    best_name, (best_model, best_acc) = max(results.items(), key=lambda kv: kv[1][1])
    print(f"Best model: {best_name} (acc={best_acc:.4f})")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.joblib')
    vec_path = os.path.join(model_dir, 'vectorizer.joblib')
    save_model_and_vectorizer(best_model, vectorizer, model_path, vec_path)
    print(f"Saved model to {model_path} and vectorizer to {vec_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save model and vectorizer')
    args = parser.parse_args()
    train(args.data_path, args.model_dir)
