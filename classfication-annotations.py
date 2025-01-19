# -----------------------------------------------------------
# Code Purpose:
# This code trains a binary classification model using a neural network
# on the provided training data and evaluates it using both validation and test datasets.
# It also generates predictions for a sample file and visualizes a confusion matrix for the sample data.
#
# Required Input Files:
# 1. 'hetero_lattice_Allen_010_diff_06_20221205-label.csv' - training dataset (contains features and labels).
#
# Output Files:
# 1. 'predicted_class.csv' - filtered samples predicted as class 1.
#
# Dependencies:
# pandas, numpy, scikit-learn, tensorflow, matplotlib
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the training data
# 'train' dataset contains features and labels, with 'label' as the target column
train = pd.read_csv('./hetero_lattice_Allen_010_diff_06_20221205-label.csv')

# Separate the features (X) and target labels (y)
X = train.drop(['tags', 'label'], axis=1)   # Drop the label column to get features
y = train['label']  # Extract the 'label' column as the target variable

# Split the data into training and validation sets (80% for training, 20% for validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# Remove non-numeric columns and keep only numeric features
X_train = X_train.select_dtypes(include=['number'])
X_val = X_val.select_dtypes(include=['number'])

# Standardize the feature data
# StandardScaler will normalize the data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_val_scaled = scaler.transform(X_val)  # Transform the validation data using the fitted scaler

# Create the neural network model
model = Sequential()

# Input layer: First layer of the neural network, with 128 neurons and ReLU activation
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

# Hidden layers: Two additional layers with 64 and 32 neurons, both using ReLU activation
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

# Output layer: 1 neuron with sigmoid activation for binary classification (0 or 1)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer, binary cross-entropy loss function, and accuracy as the metric
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training data and validate it on the validation data
# The model will train for 50 epochs with a batch size of 32
history = model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, validation_data=(X_val_scaled, y_val))

# Make predictions on the training data using the trained model
y_train_pred = model.predict(X_train_scaled)

# Round the predicted values to either 0 or 1, representing the two classes
y_train_pred = np.round(y_train_pred).astype(int)

# Filter out the samples that are predicted as class 1
# This selects all rows from X_train where the prediction is 1
predicted_class_1 = X_train[y_train_pred.flatten() == 1]

# Save the samples predicted as class 1 to a CSV file
predicted_class_1.to_csv('predicted_class.csv', index=False)

# At the end, the code will save a new CSV file containing all the samples predicted as class 1
