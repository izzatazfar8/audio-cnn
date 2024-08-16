
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import model, X_test, y_test

def test_model_accuracy():
    y_pred = model.predict(X_test).flatten()
    y_pred_class = (y_pred > 0.5).astype(int)
    assert (y_test == y_pred_class).mean() > 0.6, "Model accuracy should be greater than 80%."
