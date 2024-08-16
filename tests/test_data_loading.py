import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import extract_features

#New Recorded file tested here...

def test_data_loading():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
    features = extract_features(file_path)
    assert features.shape[1] == 13, "The number of MFCCs should be 13."
