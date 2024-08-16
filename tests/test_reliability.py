import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import model,predict_reference_sound, max_length

def test_model_reliability():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
    result_1 = predict_reference_sound(model, file_path, max_length=max_length)
    result_2 = predict_reference_sound(model, file_path, max_length=max_length)
    assert result_1 == result_2, "The model should produce consistent results on the same input."
