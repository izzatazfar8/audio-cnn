# Audio Reference Sound Detection using CNN

## Project Overview

This project, contained within the `audio-cnn` directory, aims to detect the presence of a reference sound in newly recorded audio files using a Convolutional Neural Network (CNN). The system is trained on a custom dataset of audio files and can evaluate whether a reference sound is present in a new recording by extracting relevant audio features and applying the trained model.

## Project Structure

- **`audio-cnn/`**: Root directory containing all project files.
  - **`model.py`**: Main script for training the CNN model and making predictions based on new audio files.
  - **`recorded/`**: Datasets for all of reference sound.
  - **`noise/`**: Datasets for all of noise sound.
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
  - **`report/`**: Test report generated via html,png & json.

## Installation and Setup

To use this project, you'll need to have Python 3.6 or higher installed. Follow the steps below to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/audio-cnn.git
   cd audio-cnn


## Project Summary: AI-Based Audio Reference Sound Detection
Project Overview:
This project is focused on developing and testing a Convolutional Neural Network (CNN) model that detects the presence of a reference sound in recorded audio files. The AI model is trained to distinguish between audio clips containing the reference sound and those that do not, making it useful for applications such as automated audio monitoring, quality assurance in sound production, and noise filtering.

## Key Features of the AI Model:
Model Type: The project uses a Convolutional Neural Network (CNN), a deep learning model well-suited for processing and analyzing grid-like data, such as images or, in this case, 2D representations of audio features.
Audio Features: The model extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files, which are widely used in audio processing to capture the timbral aspects of sound.
Binary Classification: The model is trained to perform binary classification, determining whether the reference sound is present (1) or not (0) in the input audio file.
Data Processing:
Feature Extraction: Audio files are processed to extract MFCC features, which are then padded or truncated to ensure consistent input dimensions.
Data Normalization: The extracted features are normalized to ensure that the model performs optimally during training.

## Model Architecture:
Layers:
Conv2D Layers: Convolutional layers extract spatial features from the MFCCs.
MaxPooling2D Layers: Pooling layers reduce the dimensionality of the data, retaining essential features while reducing computational complexity.
Dense Layers: Fully connected layers perform the final classification.
Dropout: A dropout layer is used to prevent overfitting by randomly dropping units during training.

## The project includes a comprehensive suite of tests using pytest to ensure the robustness and reliability of the AI model across multiple dimensions:

Functionality: Verifying that the model trains correctly and produces valid predictions.
Integration: Ensuring that all components (data processing, model, and prediction) work together seamlessly.
Performance: Ensuring the model trains within a reasonable timeframe.
Reliability: Confirming that the model's weights are updated after training, ensuring learning.
Regression: Ensuring model performance does not degrade after retraining.
Optimization: Testing the model with different optimizers and configurations.
Model Accuracy: Ensuring the model achieves reasonable accuracy.
Underfitting/Overfitting: Monitoring the difference between training and validation accuracy to detect issues.
Security: Ensuring the model files are saved securely and are readable.
Data Loading and Normalization: Verifying that data is loaded correctly and normalized properly.
Negative Testing: Ensuring the model handles invalid data gracefully.
Unit Testing: Verifying individual functions and components of the project.

## Challenges and Solutions:
Data Size: The project initially faced challenges related to the small dataset size. To mitigate this, data augmentation techniques such as adding noise and time-shifting were suggested.
Model Overfitting: The project included tests to detect overfitting, where the model might perform well on training data but poorly on unseen data. Solutions like dropout, L2 regularization, and model simplification were suggested.
Model Generalization: Cross-validation and hyperparameter tuning were recommended to ensure the model generalizes well across different data samples.

## Applications and Use Cases:
This AI model can be applied in various fields, including:

1. Audio Quality Assurance: Automatically detecting the presence of specific sounds in audio files for quality control in production environments.
2. Noise Monitoring: Identifying and filtering out unwanted noise in real-time audio streams.
3. Automated Audio Tagging: Automatically tagging audio files based on the presence of specific reference sounds.

