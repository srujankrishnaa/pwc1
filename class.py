import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import load_model
import shap
import tensorflow as tf
from numpy.random import seed

# Print SHAP version for debugging
import shap
print(f"SHAP version: {shap.__version__}")

# Set random seeds for reproducibility
seed(123)
tf.random.set_seed(123)

# Load and prepare data
dataset = pd.read_csv("dataset.csv")
print(dataset.head(10))

X = dataset.drop(["y"], axis=1)
Y = dataset["y"]

# Split data (must match training split)
X_train, X_other, Y_train, Y_other = train_test_split(X, Y, test_size=0.5, random_state=123)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_other, Y_other, test_size=0.5, random_state=123)

# Load model
model = load_model("model_cyclicalLR.h5")

# SHAP analysis - use a small background dataset
background = X_train.iloc[:50].values
explainer = shap.DeepExplainer(model, background)

# Get SHAP values for a few test examples
test_sample = X_test.iloc[:100].values  # Use fewer samples for testing
shap_values = explainer.shap_values(test_sample)

# Print raw SHAP values information
print(f"Type of shap_values: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"Length of shap_values list: {len(shap_values)}")
    print(f"Shape of shap_values[0]: {shap_values[0].shape}")
    # Convert to numpy array - list format is common for classification models
    shap_values = np.array(shap_values[0])
else:
    print(f"Shape of shap_values: {shap_values.shape}")

# Check shape and fix if needed
print(f"Current shape: {shap_values.shape}")
if len(shap_values.shape) == 3 and shap_values.shape[2] == 1:
    # If shape is (samples, features, 1), reshape to (samples, features)
    shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])
    print(f"Reshaped to: {shap_values.shape}")
elif len(shap_values.shape) == 2 and shap_values.shape[1] == 1:
    # If this is for a single instance with shape (features, 1)
    shap_values = shap_values.flatten()
    print(f"Flattened to: {shap_values.shape}")

# Get expected value properly
if isinstance(explainer.expected_value, list):
    expected_value = explainer.expected_value[0]
else:
    expected_value = explainer.expected_value
print(f"Expected value: {expected_value}")

# Simplified method for individual instance plot - index is relative to the test_sample
def plot_individual(index=0):
    # Get values for a single instance
    instance_values = shap_values[index]
    
    # Ensure instance_values is 1D
    if len(instance_values.shape) > 1:
        instance_values = instance_values.flatten()
    
    print(f"Instance shape: {instance_values.shape}")
    
    # Create the explanation object with minimal arguments
    explanation = shap.Explanation(
        values=instance_values,
        base_values=float(expected_value),
        data=X_test.iloc[index].values,
        feature_names=list(X_test.columns)
    )
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_instance_{index}.png")
    print(f"Saved plot for instance {index}")

# Try plotting for a couple of instances
try:
    plot_individual(0)
except Exception as e:
    print(f"Error plotting instance 0: {e}")

try:
    plot_individual(3)
except Exception as e:
    print(f"Error plotting instance 3: {e}")

# Create a summary plot
try:
    plt.figure(figsize=(10, 12))
    # Use the original X_test values corresponding to our test_sample
    shap.summary_plot(
        shap_values[:100],  # Match the number of samples we used
        X_test.iloc[:100].values,
        feature_names=list(X_test.columns),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    print("Saved summary plot")
except Exception as e:
    print(f"Error creating summary plot: {e}")

plt.show()
