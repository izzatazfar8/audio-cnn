import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import history

def test_underfitting_overfitting():
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    # Check for underfitting (high training and validation loss)
    assert training_loss[-1] < 0.5, "Model is underfitting: Training loss is too high"
    assert validation_loss[-1] < 0.5, "Model is underfitting: Validation loss is too high"
    
    # Check for overfitting (training loss much lower than validation loss)
    assert abs(training_loss[-1] - validation_loss[-1]) < 0.2, "Model is overfitting: Training loss is much lower than validation loss"

