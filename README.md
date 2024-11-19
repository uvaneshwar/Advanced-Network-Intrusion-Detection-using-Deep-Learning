Project Overview
This project implements a hybrid model to detect anomalies in network traffic data using a combination of LSTM (Long Short-Term Memory) for sequence classification and Autoencoder for anomaly detection based on reconstruction errors. The dataset contains features such as packet lengths, protocols, traffic types, and anomaly scores, which are used to identify potential network intrusions or unusual behavior.

Key Components
LSTM Model: Trained to classify sequences of network traffic data as either normal or anomalous based on key features.
Autoencoder: Learns to reconstruct input features and detects anomalies by identifying instances with high reconstruction error.
Ensemble Anomaly Detection: Combines the predictions of both the LSTM and Autoencoder models for robust anomaly detection.
Visualization: Generates insights such as reconstruction error distributions and model performance metrics (e.g., accuracy, precision, and confusion matrix).

Workflow

Data Preprocessing:
Select relevant features from the dataset.
Standardize numerical features using StandardScaler.
Convert data into sequences for the LSTM model.

LSTM Model:
Train an LSTM model to classify sequences as normal or anomalous.
Output a probability score for each sequence.

Autoencoder:
Train an autoencoder to reconstruct input data.
Compute reconstruction error to detect anomalies.

Anomaly Detection:
Use a threshold on LSTM outputs to classify anomalies.
Detect anomalies from reconstruction error using another threshold.
Combine both outputs using an ensemble approach.

Evaluation & Visualization:
Evaluate the model using accuracy, precision, and confusion matrix.
Visualize reconstruction errors and thresholds.
