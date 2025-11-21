import os
import tempfile
import pandas as pd
from src.utils import get_vectorizer, save_model_and_vectorizer, load_model_and_vectorizer, clean_text
from sklearn.linear_model import LogisticRegression


def test_clean_text():
    s = "Hello WORLD! Visit https://example.com."
    c = clean_text(s)
    assert 'https' not in c
    assert c.islower()


def test_vectorizer_and_model_save_load():
    vec = get_vectorizer(max_features=100)
    docs = ["this is real news", "fake news about something"]
    X = vec.fit_transform(docs)
    model = LogisticRegression(max_iter=200)
    model.fit(X, [1, 0])

    with tempfile.TemporaryDirectory() as d:
        mpath = os.path.join(d, 'm.joblib')
        vpath = os.path.join(d, 'v.joblib')
        save_model_and_vectorizer(model, vec, mpath, vpath)
        mdl, v = load_model_and_vectorizer(mpath, vpath)
        X2 = v.transform(["this news is real"])
        p = mdl.predict(X2)
        assert p[0] in (0,1)
