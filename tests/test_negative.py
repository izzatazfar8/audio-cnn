
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import model, predict_reference_sound, max_length

def test_negative_cases():
    # Test cases where the reference sound should not be present
    non_reference_files = [
        #'C:/Users/izulhish/Downloads/audio-cnn/noise/noise.wav',
        #'C:/Users/izulhish/Downloads/audio-cnn/noise/radio-electro-glitch.wav',
        'C:/Users/izulhish/Downloads/audio-cnn/noise/recorded-totally-noise.wav',
    ]
    
    for file_path in non_reference_files:
        is_present = predict_reference_sound(model, file_path, max_length=max_length)
        assert not is_present, f"Failed for file: {file_path}"

