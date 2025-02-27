import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import load_model
import shap
import tensorflow as tf
from numpy.random import seed

# Print versions for debugging
import shap
print(f"SHAP version: {shap.__version__}")
print(f"TensorFlow version: {tf.__version__}")

# Set random seeds for reproducibility
seed(123)
tf.random.set_seed(123)

# Load and prepare data
dataset = pd.read_csv("dataset.csv")
print(dataset.head(5))

X = dataset.drop(["y"], axis=1)
Y = dataset["y"]

# Split data (must match training split)
X_train, X_other, Y_train, Y_other = train_test_split(X, Y, test_size=0.5, random_state=123)
X_validation, X_test, Y_validation, Y_test = train_test_split(X_other, Y_other, test_size=0.5, random_state=123)

# Load model
model = load_model("model_cyclicalLR.h5")

# SHAP analysis - use a small background dataset
print("Creating SHAP explainer...")
background = X_train.iloc[:50].values
explainer = shap.DeepExplainer(model, background)

# Get SHAP values
print("Calculating SHAP values...")
test_sample = X_test.iloc[:500].values  # Use fewer samples for faster analysis
shap_values = explainer.shap_values(test_sample)

# Process SHAP values
if isinstance(shap_values, list):
    print(f"Shape of first element in shap_values list: {shap_values[0].shape}")
    # For binary classification, we typically use the first element
    shap_values = shap_values[0]
else:
    print(f"Shape of shap_values: {shap_values.shape}")

# Make sure we have a 2D array of (samples, features)
if len(shap_values.shape) == 3:
    shap_values = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])
    print(f"Reshaped to: {shap_values.shape}")

# Calculate feature importance (mean absolute SHAP value for each feature)
feature_importance = np.abs(shap_values).mean(axis=0)
feature_names = list(X_test.columns)

# Create a DataFrame for easy sorting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('Importance', ascending=False)

# Print top 10 most important features
print("\n=== Top 10 Most Important Features ===")
print(importance_df.head(10))

# ------- VISUALIZATIONS -------

# 1. Bar chart of feature importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Mean |SHAP value|')
plt.title('Top 15 Most Important Features')
plt.gca().invert_yaxis()  # Display highest importance at the top
plt.tight_layout()
plt.savefig("top_features_bar.png")
print("Saved top features bar chart")

# 2. SHAP summary plot (better visualization of feature importance)
try:
    plt.figure(figsize=(12, 10))
    # Only show top 15 features for clarity
    top_indices = [feature_names.index(feature) for feature in top_features['Feature']]
    
    # Extract data for top features
    top_shap_values = shap_values[:, top_indices]
    top_feature_names = top_features['Feature'].tolist()
    top_feature_data = X_test.iloc[:500][top_feature_names].values
    
    shap.summary_plot(
        top_shap_values, 
        top_feature_data,
        feature_names=top_feature_names,
        show=False,
        plot_type="bar"  # Bar chart focused on feature importance
    )
    plt.tight_layout()
    plt.savefig("shap_importance_summary.png")
    print("Saved SHAP importance summary plot")
except Exception as e:
    print(f"Error creating summary plot: {e}")

# 3. SHAP summary dot plot (shows direction of feature effects)
try:
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        top_shap_values, 
        top_feature_data,
        feature_names=top_feature_names,
        show=False,
        plot_type="dot"  # Dot plot shows impact direction (positive/negative)
    )
    plt.tight_layout()
    plt.savefig("shap_dot_summary.png")
    print("Saved SHAP dot summary plot")
except Exception as e:
    print(f"Error creating dot summary plot: {e}")

# 4. SHAP dependence plots for top 3 features
try:
    for i, feature in enumerate(top_features['Feature'][:3]):
        feature_idx = feature_names.index(feature)
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            feature_idx, 
            shap_values, 
            X_test.iloc[:500].values,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Dependence Plot for {feature}")
        plt.tight_layout()
        plt.savefig(f"dependence_plot_{i}_{feature.replace(' ', '_')}.png")
        print(f"Saved dependence plot for {feature}")
except Exception as e:
    print(f"Error creating dependence plots: {e}")

# Display all plots
plt.show()

# Save the feature importance data to CSV
importance_df.to_csv("feature_importance.csv", index=False)
print("Saved feature importance to feature_importance.csv")