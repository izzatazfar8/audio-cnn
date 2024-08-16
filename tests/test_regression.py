import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import model,predict_reference_sound, model, max_length

def test_regression():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
    result = predict_reference_sound(model, file_path, max_length=max_length)
    expected_output = True  # Assuming the correct output for this file is known to be True
    assert result == expected_output, "Regression: Model's output should match the expected output after changes."
