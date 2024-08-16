# Audio Reference Sound Detection using CNN

## Project Overview

This project, contained within the `audio-cnn` directory, aims to detect the presence of a reference sound in newly recorded audio files using a Convolutional Neural Network (CNN). The system is trained on a custom dataset of audio files and can evaluate whether a reference sound is present in a new recording by extracting relevant audio features and applying the trained model.

## Project Structure

- **`audio-cnn/`**: Root directory containing all project files.
  - **`model.py`**: Main script for training the CNN model and making predictions based on new audio files.
  - **`tests/`**: Directory containing all `pytest` test scripts for validating various testing aspects of the project.
    - **`tests/test_functionality.py`**: Ensures that the main functionalities of the system work as expected.
    - **`tests/test_integration.py`**: Verifies the integration between different components of the system.
    - **`tests/test_performance.py`**: Tests the performance and efficiency of the model under various conditions.
    - **`tests/test_reliability.py`**: Assesses the system's reliability by testing it under different scenarios.
    - **`tests/test_regression.py`**: Ensures that new changes do not break existing functionality.
    - **`tests/test_model_accuracy.py`**: Validates the accuracy of the CNN model against the test dataset.
    - **`tests/test_data_loading.py`**: Checks if the dataset is loaded correctly and is in the expected format.
    - **`tests/test_negative.py`**: Tests cases where the reference sound should not be detected.
    - **`tests/test_unit.py`**: Unit tests for verifying individual components of the system.
    - **`tests/test_underfitting_overfitting.py`**: Checks for signs of underfitting or overfitting by analyzing training and validation losses.
    - **`tests/test_security.py`**: Ensures that the recorded files are secure and correctly formatted.
  - **`requirements.txt`**: Contains all the required Python libraries needed to run this project.
  - **`README.md`**: Documentation for the project, including setup instructions, usage, and contribution guidelines.

## Installation and Setup

To use this project, you'll need to have Python 3.6 or higher installed. Follow the steps below to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/audio-cnn.git
   cd audio-cnn

Training the Model:
Run the main script to train the CNN model using your custom dataset:
python model.py
Running Tests:
Execute the pytest tests to validate the functionality, performance, and reliability of the system:
pytest --html=report.html
Making Predictions:
Use the trained model to predict whether a reference sound is present in a new recording by running:
python predict_reference_sound.py <path_to_audio_file>
