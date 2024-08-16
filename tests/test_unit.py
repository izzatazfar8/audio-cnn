import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import extract_features, predict_reference_sound, max_length, model
import numpy as np
from sklearn.model_selection import train_test_split

def test_feature_extraction():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/16bits.wav'
    features = extract_features(file_path)
    
    # Ensure that features are extracted and have the correct shape
    assert features is not None, "Feature extraction returned None"
    assert isinstance(features, np.ndarray), "Extracted features should be a NumPy array"
    assert features.shape[1] == 13, "The number of MFCC features should be 13"


def test_labels_are_boolean():
    # Redefine the labels used in the main code
    labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  # Corresponding labels (1 for reference sound, 0 for non-reference sound)
    
    # Check that the labels in your dataset are boolean (0 or 1)
    unique_labels = np.unique(labels)
    assert set(unique_labels).issubset({0, 1}), "Labels should only contain 0 or 1"

def test_model_structure():
    # Check if the model layers are correctly structured
    assert len(model.layers) > 0, "Model should have at least one layer"
    assert model.input_shape == (None, max_length, 13, 1), "Model input shape should match the input data shape"
    assert model.output_shape == (None, 1), "Model output shape should be a single neuron"