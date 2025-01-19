# -----------------------------------------------------------
# Code Purpose: 
# This code trains a binary classification model using a neural network 
# on the provided training data and evaluates it using both validation and test datasets.
# It also generates predictions for a sample file and visualizes a confusion matrix for the sample data.
#
# Required Input Files:
# 1. 'hetero_lattice_Allen_010_diff_06_20240514_tmds.csv' - training dataset (contains features and labels).
# 2. 'hetero_lattice_Allen_010_diff_06_20221205.csv' - testing dataset (contains features).
# 3. 'sample.csv' - sample data file (for confusion matrix visualization).
#
# Output Files:
# 1. 'predicted_results.csv' - test data predictions with the predicted labels.
# 2. 'confusion_matrix_sample.csv' - confusion matrix for the sample data.
#
# Dependencies:
# pandas, numpy, scikit-learn, tensorflow, matplotlib, seaborn
# -----------------------------------------------------------

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------
# Load the datasets
# 'train' contains both features and labels for training.
# 'test' contains only features for testing.
# 'sample_file' contains sample data for confusion matrix visualization.
# -----------------------------------------------------------
train = pd.read_csv('./hetero_lattice_Allen_010_diff_06_20240514_tmds.csv')
test = pd.read_csv('./hetero_lattice_Allen_010_diff_06_20221205.csv')
sample_file = pd.read_csv('./sample.csv')  

# Separate the features (X) and the target labels (y) for the training dataset
X = train.drop(['tags', 'label'], axis=1)  # Dropping non-feature columns: 'tags' and 'label'
y = train['label']  # 'label' is the target variable (0 or 1)
X_test = test.drop(['tags'], axis=1)  # Dropping 'tags' column from test data as it is not a feature

# Split the training data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

# Scale the data using StandardScaler for normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on the training data
X_val_scaled = scaler.transform(X_val)  # Transform the validation data using the fitted scaler
X_test_scaled = scaler.transform(X_test)  # Transform the test data

# -----------------------------------------------------------
# Build the neural network model using Keras Sequential API
# The model is a fully connected feedforward neural network.
# Input layer, hidden layers, and output layer (sigmoid for binary classification).
# -----------------------------------------------------------
model = Sequential()

# Input layer with 128 neurons and ReLU activation function
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Hidden layers with 64 and 32 neurons respectively, with ReLU activation and dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Output layer with 1 neuron, using sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------------------------------------
# Train the model
# The model is trained on the scaled training data (X_train_scaled) with labels (y_train)
# Validation data is used to monitor performance during training.
# -----------------------------------------------------------
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_val_scaled, y_val))

# -----------------------------------------------------------
# Evaluate the model performance on the validation set
# Predictions are made for the validation data, and performance metrics are calculated.
# -----------------------------------------------------------
y_val_pred = model.predict(X_val_scaled)
y_val_pred = np.round(y_val_pred).astype(int)  # Convert probabilities to binary (0 or 1)

# Calculate accuracy and display classification report
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {accuracy}")
print(classification_report(y_val, y_val_pred))

# -----------------------------------------------------------
# Generate predictions for the test dataset
# The test data does not have labels, so we predict the labels for it.
# The results are saved in 'predicted_results.csv'.
# -----------------------------------------------------------
y_test_pred = model.predict(X_test_scaled)
y_test_pred = np.round(y_test_pred).astype(int)

test['predicted_label'] = y_test_pred  # Add predicted labels to the test dataframe
test.to_csv('predicted_results.csv', index=False)  # Save the predictions to a CSV file

# -------- Adding confusion matrix for the sample file --------
# This part evaluates the performance of the model on the sample dataset.
# A confusion matrix is generated and visualized.
# -----------------------------------------------------------
X_sample = sample_file.drop(['tags', 'label'], axis=1)  # Features from the sample file
y_sample_true = sample_file['label']  # True labels from the sample file

# Scale the sample data
X_sample_scaled = scaler.transform(X_sample)

# Make predictions on the sample data
y_sample_pred = model.predict(X_sample_scaled)
y_sample_pred = np.round(y_sample_pred).astype(int)  # Convert predictions to binary (0 or 1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_sample_true, y_sample_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Sample Data')
plt.show()

# Save the confusion matrix to a CSV file
np.savetxt('confusion_matrix_sample.csv', conf_matrix, delimiter=',', fmt='%d')