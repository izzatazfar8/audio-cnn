import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to extract features from audio file
def extract_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.T

# Function to load custom dataset
def load_custom_dataset(file_paths, labels, n_mfcc=13, max_length=174):
    X, y = [], []
    for file_path, label in zip(file_paths, labels):
        features = extract_features(file_path, n_mfcc=n_mfcc)
        if len(features) < max_length:
            features = np.pad(features, ((0, max_length - len(features)), (0, 0)), 'constant')
        else:
            features = features[:max_length, :]
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

# Example file paths and labels
file_paths = [
    'C:/Users/izulhish/Downloads/audio-cnn/recorded/16_LE_stereo_sine.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/recorded/32_LE_stereo_sine.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/noise/noise.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/noise/radio-electro-glitch.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/noise/granular-noise-crackle.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/noise/harsh-radio-static.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/noise/radio-stutter-noise.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/noise/white-noise.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/recorded/16bits.wav',
    'C:/Users/izulhish/Downloads/audio-cnn/recorded/32bits.wav'
    
]
labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  # Corresponding labels (1 for reference sound, 0 for non-reference sound)

# Load dataset
X, y = load_custom_dataset(file_paths, labels)

# Pad sequences to the same length for CNN input
max_length = max([len(x) for x in X])
X_padded = np.array([np.pad(x, ((0, max_length - len(x)), (0, 0)), 'constant') for x in X])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Reshape data for CNN input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model and capture the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Print the keys of the history to see what was recorded
print("Training completed. Available metrics in history:", history.history.keys())


# Evaluate model
y_pred = model.predict(X_test).flatten()
y_pred_class = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_class))

# Plot precision-recall curve and save as PNG
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')  # Save the plot as PNG
#plt.show()
#plt.close()

# Function to predict presence of reference sound
def predict_reference_sound(model, file_path, max_length, threshold=0.8):
    # Extract MFCC features from the audio file
    print("Step 1: Extracting MFCC features from the audio file...")
    features = extract_features(file_path)
    print(f"Extracted MFCC features shape: {features.shape}")
    print("MFCC (Mel-Frequency Cepstral Coefficients) are a representation of the short-term power spectrum of a sound, useful for distinguishing audio characteristics.")
    
    # Pad or truncate the MFCC features to a consistent length
    print("\nStep 2: Padding or truncating MFCC features to a consistent length...")
    if len(features) < max_length:
        features_padded = np.pad(features, ((0, max_length - len(features)), (0, 0)), 'constant')
    else:
        features_padded = features[:max_length, :]
    print(f"Padded MFCC features shape: {features_padded.shape}")
    print("This ensures that all audio samples have the same size input to the CNN, regardless of their original length.")
    
    # Reshape the features for CNN input
    print("\nStep 3: Reshaping features for CNN input...")
    features_padded = features_padded[np.newaxis, ..., np.newaxis]  # Reshape for CNN input
    print(f"Input shape for CNN: {features_padded.shape}")
    print("The input is reshaped to fit the CNN's expected input format.")
    
    # Predict the presence of the reference sound
    print("\nStep 4: Making prediction using the CNN model...")
    prediction = model.predict(features_padded).flatten()[0]
    print(f"Prediction value: {prediction}")
    print(f"Threshold: {threshold}")
    print("The prediction value indicates the model's confidence that the reference sound is present in the audio file. It is compared to a threshold to make a binary decision.")
    
    return prediction > threshold

# Example prediction
new_recorded_file = 'C:/Users/izulhish/Downloads/audio-cnn/recorded/recorded-sine-1.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-sine-2.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-sine-3-minor-noise.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-sine-4.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-sine-5-clean.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-sine-6-minor-noise.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-surroundings.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-glitch.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-totally-noise.wav'
#new_recorded_file = '/home/rpl2/izzat/recorded/recorded-nothing.wav'

is_present = predict_reference_sound(model, new_recorded_file, max_length=max_length)

if is_present:
    print("Final Result : The reference sound is present in the recorded file.")
else:
    print("Final Result : The reference sound is NOT present in the recorded file.")

