
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import predict_reference_sound, model, max_length

def test_reference_sound_present():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
    result = predict_reference_sound(model, file_path, max_length=max_length)
    assert result == True, "The reference sound should be detected."

def test_reference_sound_not_present():
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/noise/harsh-radio-static.wav'
    result = predict_reference_sound(model, file_path, max_length=max_length)
    assert result == False, "The reference sound should not be detected."
