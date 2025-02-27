# DeepBankPred: Predicting Term Deposit Subscriptions using Deep Learning

## Project Overview
This project implements a deep neural network (DNN) to predict whether a customer will subscribe to a term deposit. The model is trained on bank marketing data and evaluated using SHAP for interpretability.

## Features
- **Deep Learning Model**: Multi-layer perceptron (MLP) with dropout layers.
- **Cyclical Learning Rate (CLR)**: Improves training convergence.
- **Class Weighting**: Addresses data imbalance.
- **SHAP Analysis**: Provides explainability to predictions.

## Dataset
The dataset (`dataset.csv`) contains features like age, balance, job type, and contact history. The target variable (`y`) indicates subscription (`1`) or not (`0`).

## Installation
### Requirements
Ensure you have the following libraries installed:
```bash
pip install tensorflow keras numpy pandas scikit-learn shap

Training the Model
Run the following command to train the model:

bash
Copy
Edit

python train.py

The trained model is saved as model_cyclicalLR.h5.
Training history is saved as training_history.pkl.

Model Evaluation

The model evaluates accuracy and other metrics on the test set.
SHAP values are used to interpret feature contributions.
Visualizations
SHAP analysis is performed to explain model predictions:

shap_instance_0.png
shap_instance_3.png
shap_summary_plot.png

Authors

Developed by srujan krishna.

License
This project is open-source under the MIT License.

Let me know if you need any modifications! 🚀
