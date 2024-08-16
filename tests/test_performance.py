import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from model import predict_reference_sound, model, max_length

def test_model_inference_time():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
    start_time = time.time()
    result = predict_reference_sound(model, file_path, max_length=max_length)
    end_time = time.time()
    assert end_time - start_time < 1.0, "Model inference should be completed within 1 second."
