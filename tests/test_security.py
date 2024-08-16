import os, sys, pytest, wave
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_file_security():
    #new recorded file
    file_path = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
    
    # Check if the file exists
    assert os.path.exists(file_path), "Recorded file does not exist"
    
# Check if the file is a WAV file based on the extension
    assert file_path.lower().endswith('.wav'), "Recorded file is not a WAV file based on the extension"
    
    # Check if the file is a valid WAV file by attempting to open it
    try:
        with wave.open(file_path, 'rb') as wav_file:
            assert wav_file.getnchannels() > 0, "WAV file should have at least one channel"
            assert wav_file.getsampwidth() > 0, "WAV file should have a valid sample width"
            assert wav_file.getframerate() > 0, "WAV file should have a valid frame rate"
            assert wav_file.getnframes() > 0, "WAV file should have valid frames"
    except wave.Error as e:
        pytest.fail(f"File is not a valid WAV file: {e}")

