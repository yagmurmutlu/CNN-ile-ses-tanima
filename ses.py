import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

# Load audio file using librosa
audio_file_path = r"C:\Users\Lenovo\OneDrive\Masaüstü\CNN ile ses tanıma\dataset\fold1\7061-6-0-0.wav"
librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)
print(librosa_audio_data)
print(librosa_audio_data.shape)

# Plot the audio signal
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)
plt.show()

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)

# Load metadata
audio_dataset_path = "C:\\Users\\Lenovo\\OneDrive\\Masaüstü\\CNN ile ses tanıma\\dataset"
metadata = pd.read_csv("C:\\Users\\Lenovo\\OneDrive\\Masaüstü\\CNN ile ses tanıma\\metadata\\UrbanSound8K.csv")

def feature_extractor(file):
    audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    
    return mfccs_scaled_features

# Extract features from audio files
extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path), "fold" + str(row["fold"]), str(row["slice_file_name"]))
    final_class_labels = row["class"]
    data = feature_extractor(file_name)
    extracted_features.append([data, final_class_labels])

# Convert features and labels to numpy arrays
features_df = pd.DataFrame(extracted_features, columns=['feature', 'class_label'])

X = np.array(features_df.feature.tolist())
y = np.array(features_df.class_label.tolist())

# Encode the class labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state=0)

# Build the model
num_labels = yy.shape[1]
model = Sequential()

# 1st hidden layer
model.add(Dense(125, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 2nd hidden layer
model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
epoch_count = 300
num_batch_size = 32
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=epoch_count, validation_data=(x_test, y_test), verbose=1)

# Evaluate the model
validation_test_set_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(validation_test_set_accuracy[1])

filename = r"C:\Users\Lenovo\OneDrive\Masaüstü\CNN ile ses tanıma\4912-3-2-0.wav"
sound_signal,sample_rate=librosa.load(filename,res_type="kaiser_fast")
mfcc_features=librosa.feature.mfcc(y=sound_signal,sr=sample_rate,n_mfcc=40)
mfccs_scaled_features=np.mean(mfcc_features.T,axis=0)

result_array=model.predict(mfccs_scaled_features)
result_classes=["klime,korna,çocuk sesleri,köpek havlaması,sondaj,motor sesi,silah ses,darbeli matkap,siren,sokak müziği"]

result=np.argmax(result_array[0])
result_classes[result]
