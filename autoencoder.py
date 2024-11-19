import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model

# Function to simulate image loading (replace with your actual loading function)
def load_images_from_directory(directory, img_size=(28, 28)):
    # Placeholder for the sake of this example. Load your actual images here.
    # Ensure images have shape (num_samples, height, width, 1)
    return np.random.rand(39950, 28, 28, 1)  # Simulated data (39950 images of size 28x28 with 1 channel)

# Load images (replace this with the actual directory)
images = load_images_from_directory('/content/lstm_images')

# Simulate ground truth labels (e.g., 10% anomalies)
np.random.seed(42)  # For reproducibility
anomaly_ratio = 0.10
num_samples = images.shape[0]
y_true = np.zeros(num_samples, dtype=int)
anomaly_indices = np.random.choice(num_samples, size=int(num_samples * anomaly_ratio), replace=False)
y_true[anomaly_indices] = 1  # Label anomalies as 1

# Split data and labels into training and testing sets (80-20%)
train_images, test_images, y_train_true, y_test_true = train_test_split(
    images, y_true, test_size=0.2, random_state=42, stratify=y_true
)

# Load the prebuilt autoencoder model
autoencoder = load_model('/content/autoencoder_model.keras')  # Replace with your actual model path

# Get reconstructed images for both training and testing sets
train_reconstructed = autoencoder.predict(train_images)
test_reconstructed = autoencoder.predict(test_images)

# Calculate reconstruction error for training and testing sets
train_reconstruction_error = np.mean(np.abs(train_images - train_reconstructed), axis=(1, 2, 3))
test_reconstruction_error = np.mean(np.abs(test_images - test_reconstructed), axis=(1, 2, 3))

# Set a threshold for anomalies based on the 90th percentile of the training reconstruction error
threshold = np.percentile(train_reconstruction_error, 90)
print(f"Anomaly Detection Threshold: {threshold}")

# Determine anomalies in training and testing sets based on the threshold
y_train_pred = (train_reconstruction_error > threshold).astype(int)
y_test_pred = (test_reconstruction_error > threshold).astype(int)

# Print number of detected anomalies in training and testing sets
print("Number of detected anomalies in training set:", np.sum(y_train_pred))
print("Number of detected anomalies in testing set:", np.sum(y_test_pred))

# Simulate protocol groups (replace with actual protocol data)
protocol_groups_train = np.random.choice(['HTTP', 'FTP', 'SSH', 'DNS'], size=len(y_train_pred))
protocol_groups_test = np.random.choice(['HTTP', 'FTP', 'SSH', 'DNS'], size=len(y_test_pred))

# Get protocol counts for anomalies in training and testing sets
train_anomaly_protocol_counts = pd.Series(protocol_groups_train[y_train_pred == 1]).value_counts()
test_anomaly_protocol_counts = pd.Series(protocol_groups_test[y_test_pred == 1]).value_counts()

# Classification report
print("\nClassification Report for Training Set:")
print(classification_report(y_train_true, y_train_pred, target_names=["Normal", "Anomaly"]))

print("\nClassification Report for Testing Set:")
print(classification_report(y_test_true, y_test_pred, target_names=["Normal", "Anomaly"]))

# Visualization: Anomaly distribution in pie chart
def plot_anomaly_distribution(train_counts, test_counts):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    if not train_counts.empty:
        train_counts.plot.pie(ax=axes[0], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        axes[0].set_title('Training Set Anomaly Distribution by Protocol')
        axes[0].set_ylabel('')  # Hide y-label

    else:
        axes[0].pie([1], labels=["No Anomalies Detected"], colors=["gray"], autopct='%1.1f%%')
        axes[0].set_title('Training Set Anomaly Distribution by Protocol')

    if not test_counts.empty:
        test_counts.plot.pie(ax=axes[1], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        axes[1].set_title('Testing Set Anomaly Distribution by Protocol')
        axes[1].set_ylabel('')  # Hide y-label
    else:
        axes[1].pie([1], labels=["No Anomalies Detected"], colors=["gray"], autopct='%1.1f%%')
        axes[1].set_title('Testing Set Anomaly Distribution by Protocol')

    plt.show()

plot_anomaly_distribution(train_anomaly_protocol_counts, test_anomaly_protocol_counts)
# Print protocol counts for anomalies in training and testing sets
print("Anomaly Protocol Counts in Training Set:")
print(train_anomaly_protocol_counts)

print("\nAnomaly Protocol Counts in Testing Set:")
print(test_anomaly_protocol_counts)

# Visualization: Anomaly distribution in pie chart
plot_anomaly_distribution(train_anomaly_protocol_counts, test_anomaly_protocol_counts)

