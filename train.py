import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
from numpy.random import seed
from sklearn.utils import class_weight

# Custom modules
from CLR import CyclicLR
# Uncomment the next line if you created metrics.py
# from metrics import *

# Set random seeds for reproducibility
seed(123)
tf.random.set_seed(123)

# Load and prepare data
dataset = pd.read_csv("dataset.csv")
print(dataset.head(10))

X = dataset.drop(["y"], axis=1)
Y = dataset["y"]

# Split data
X_train, X_other, Y_train, Y_other = train_test_split(X, Y, test_size=0.5, random_state=123)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_other, Y_other, test_size=0.5, random_state=123)

# Create DNN model
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Initialize model
model = create_model(input_dim=X_train.shape[1])

# Compile model
# If using metrics.py, include custom metrics; otherwise, use just 'accuracy'
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])  # Add specificity, precision, etc., if metrics.py exists

# Define cyclical learning rate callback
clr = CyclicLR(
    base_lr=0.001,
    max_lr=0.006,
    step_size=2000.,
    mode='triangular'
)

# Calculate class weights using sklearn
classes = np.unique(Y_train)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=Y_train
)
class_weight_dict = dict(zip(classes, class_weights))

# Train the model
history = model.fit(
    X_train.values,
    Y_train.values,
    epochs=50,
    batch_size=32,
    validation_data=(X_validation.values, Y_validation.values),
    callbacks=[clr],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate on test set
test_results = model.evaluate(
    X_test.values,
    Y_test.values,
    verbose=0,
    return_dict=True
)
print(f"\nTest Results:")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# Save the trained model
model.save("model_cyclicalLR.h5")
print("Model saved as 'model_cyclicalLR.h5'")

# Optional: Save training history
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved as 'training_history.pkl'")