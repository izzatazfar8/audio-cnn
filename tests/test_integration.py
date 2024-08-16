import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import load_custom_dataset, model, max_length

def test_pipeline():
    file_paths = [
        'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav',
        'C:/Users/izulhish/Downloads/audio-cnn/noise/harsh-radio-static.wav',
    ]
    labels = [1, 0]
    X, y = load_custom_dataset(file_paths, labels)
    X_padded = np.array([np.pad(x, ((0, max_length - len(x)), (0, 0)), 'constant') for x in X])
    X_padded = X_padded[..., np.newaxis]
    predictions = model.predict(X_padded).flatten()
    assert len(predictions) == len(file_paths), "The number of predictions should match the number of input files."
