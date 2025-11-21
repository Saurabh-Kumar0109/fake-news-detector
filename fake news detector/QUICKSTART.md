# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 2. Run Demo
```cmd
python demo.py
```
This will:
- Generate 500 sample news articles (fake & real)
- Train both Logistic Regression and SVM models
- Create evaluation plots (confusion matrices, ROC curves)
- Make predictions on example texts

### 3. View Results
Check `models_demo\artifacts\` for:
- `logreg_confusion_matrix.png` - Confusion matrix visualization
- `logreg_roc_curve.png` - ROC curve showing model performance
- `logreg_classification_report.txt` - Detailed metrics

## 📊 Expected Output

The demo will show predictions like:
- ✅ **REAL**: "City council approves new infrastructure project"
- ❌ **FAKE**: "SHOCKING: Coffee cures cancer overnight!"

## 🧪 Try Your Own Predictions

```cmd
python src\predict.py --model-path models\best_model.joblib --text "Your news headline here"
```

## 📓 Explore Interactively

```cmd
jupyter notebook notebooks\exploration.ipynb
```

## 🎯 Use Real Data

1. Download Kaggle "Fake News Dataset"
2. Place at `data\train.csv`
3. Train: `python src\train.py --data-path data\train.csv --model-dir models`

## 🛠️ Project Features

✅ TF-IDF text vectorization  
✅ Logistic Regression classifier  
✅ SVM (Linear SVC) classifier  
✅ Automatic best model selection  
✅ Confusion matrix visualization  
✅ ROC curve analysis  
✅ Sample data generator  
✅ Jupyter notebook for EDA  
✅ CLI for predictions  
✅ Unit tests  

## 📁 What Gets Created

```
models/
├── best_model.joblib           # Trained classifier
├── vectorizer.joblib           # TF-IDF vectorizer
└── artifacts/
    ├── logreg_confusion_matrix.png
    ├── logreg_roc_curve.png
    ├── logreg_classification_report.txt
    ├── svm_confusion_matrix.png
    ├── svm_roc_curve.png
    └── svm_classification_report.txt
```

## 🔬 Model Performance

With synthetic data (200-1000 samples):
- **Accuracy**: ~100% (template-based patterns are easy to learn)
- **Real-world data**: Expect 85-95% accuracy on Kaggle dataset

## 💡 Tips

1. **More data = Better results**: Generate more samples with `--n-samples 5000`
2. **Check artifacts**: Visual plots help understand model behavior
3. **Experiment**: Try changing TF-IDF parameters in `src/utils.py`
4. **Real dataset**: Use Kaggle data for production-quality results

## 🐛 Troubleshooting

**Import errors?**
- Make sure you're in the project root directory
- Scripts use `sys.path.insert(0, ...)` to handle imports

**Dependencies missing?**
- Run: `pip install -r requirements.txt`

**Want to retrain?**
- Delete `models/` folder and run training again

## 📚 Learn More

- Read the full [README.md](README.md) for detailed documentation
- Explore [notebooks/exploration.ipynb](notebooks/exploration.ipynb) for step-by-step analysis
- Check [tests/test_pipeline.py](tests/test_pipeline.py) for usage examples
