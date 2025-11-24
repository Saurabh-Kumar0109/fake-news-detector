"""
Demo script to showcase the Fake News Detection project.
This script generates data, trains models, and makes predictions.
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("FAKE NEWS DETECTION - DEMO")
print("=" * 60)

# Step 1: Generate sample data
print("\n[1/4] Generating sample dataset...")
from src.generate_sample_data import generate_dataset
train_path, test_path = generate_dataset(500, 'data/demo_data.csv', test_split=0.2)
print(f"✓ Created {train_path} and {test_path}")

# Step 2: Train models
print("\n[2/4] Training models...")
from src.train import train
train(train_path, 'models_demo', test_size=0.2)
print("✓ Models trained and saved to models_demo/")

# Step 3: Load model and make predictions
print("\n[3/4] Loading trained model...")
from src.utils import load_model_and_vectorizer, clean_text
model, vectorizer = load_model_and_vectorizer(
    'models_demo/best_model.joblib',
    'models_demo/vectorizer.joblib'
)
print("✓ Model loaded")

# Step 4: Make predictions
print("\n[4/4] Making predictions on sample texts...")
print("=" * 60)

test_texts = [
    "SHOCKING: Coffee cures cancer overnight!",
    "City council approves new infrastructure project",
    "You won't believe what this celebrity said!",
    "Study published in Nature examines climate effects",
    "BREAKING: Government secretly planning to control internet!",
    "Company reports quarterly earnings, stock rises by 5%"
]
while True:
    a=input("Enter any text to test the model (or press Enter to skip): ")
    cleaned = clean_text(a)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    label = "🔴 FAKE" if pred == 0 else "🟢 REAL"
    '''for i, text in enumerate(list(a), 1):
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        label = "🔴 FAKE" if pred == 0 else "🟢 REAL"
    '''
    print(f"\n. {a}")
    print(f"   → Prediction: {label}")

    print("\n" + "=" * 60)
    print("Demo complete! Check models_demo/artifacts/ for plots.")
    print("=" * 60)
