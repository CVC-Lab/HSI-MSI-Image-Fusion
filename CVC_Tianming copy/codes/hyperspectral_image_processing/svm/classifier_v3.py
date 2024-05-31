from scipy.io import loadmat
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd  # For generating tables


def load_data():
    data = loadmat('../Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = loadmat('../Indian_pines_gt.mat')['indian_pines_gt']
    # Load the data
    data_fused = loadmat('../gt&SRI.mat')['SRI']
    labels_fused = loadmat('../gt&SRI.mat')['gt']
    return data, data_fused, labels, labels_fused

def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics_report = classification_report(y_test, y_pred, output_dict=True)
    return y_pred, metrics_report

def visualize_data(data, labels):
    plt.subplot(1, 2, 1)
    plt.imshow(data[:, :, 29], cmap='gray')
    plt.title("Processed Hyperspectral Image (one band)")

    plt.subplot(1, 2, 2)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title("Ground Truth")
    plt.show()

if __name__ == "__main__":
    # Load data
    original_data, fused_data, labels, labels_fused = load_data()
    
    # Check if the original and fused datasets are of the same shape
    if original_data.shape != fused_data.shape or labels.shape != labels_fused.shape:
        print("Warning: The shapes of the original and fused datasets do not match.")
    
    # Reshape original and fused data for model training
    original_data_2d = original_data.reshape((-1, original_data.shape[-1]))
    fused_data_2d = fused_data.reshape((-1, fused_data.shape[-1]))

    labels_2d = labels.reshape(-1)
    labels_fused_2d = labels_fused.reshape(-1)

    # Split data for the original dataset
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(original_data_2d, labels_2d, test_size=0.3, stratify=labels_2d, random_state=42)
    
    # Split data for the fused dataset
    X_train_fused, X_test_fused, y_train_fused, y_test_fused = train_test_split(fused_data_2d, labels_fused_2d, test_size=0.3, stratify=labels_fused_2d, random_state=42)

    # Train model and get metrics on original data
    _, metrics_orig = train_and_evaluate(X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    
    # Train model and get metrics on fused data
    _, metrics_fused = train_and_evaluate(X_train_fused, y_train_fused, X_test_fused, y_test_fused)

    # Compare performance metrics
    print(f"Accuracy on original data: {metrics_orig['accuracy']}")
    print(f"Accuracy on fused data: {metrics_fused['accuracy']}")
    
    print(f"F1-Score (weighted) on original data: {metrics_orig['weighted avg']['f1-score']}")
    print(f"F1-Score (weighted) on fused data: {metrics_fused['weighted avg']['f1-score']}")
