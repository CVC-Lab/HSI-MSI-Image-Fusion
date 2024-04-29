from scipy.io import loadmat
import numpy as np
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd  # For generating tables


def load_data():
    # Load original data
    data = loadmat('../Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = loadmat('../Indian_pines_gt.mat')['indian_pines_gt']
    
    # Load fused data
    data_fused = loadmat('../gt&SRI.mat')['SRI']
    labels_fused = loadmat('../gt&SRI.mat')['gt']

    # Load MSI data
    data_msi = loadmat('../HSI_MSI.mat')['MSI']
    labels_msi = loadmat('../gt&SRI.mat')['gt']

    return data, data_fused, labels, labels_fused, data_msi, labels_msi  # Now returning 6 items


def train_and_evaluate(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics_report = classification_report(y_test, y_pred, output_dict=True)
    return y_pred, metrics_report


def visualize_data(data, labels):
    plt.subplot(1, 2, 1)
    plt.imshow(data[:, :, 1], cmap='gray')
    plt.title("Processed Hyperspectral Image (one band)")

    plt.subplot(1, 2, 2)
    plt.imshow(labels, cmap='nipy_spectral')
    plt.title("Ground Truth")
    plt.show()


def plot_metrics(metrics_orig, metrics_fused, metrics_msi):
    # Metrics to be compared
    metrics_names = ['Accuracy', 'F1-Score (weighted)']
    
    # Data
    orig_metrics = [metrics_orig['accuracy'], metrics_orig['weighted avg']['f1-score']]
    fused_metrics = [metrics_fused['accuracy'], metrics_fused['weighted avg']['f1-score']]
    msi_metrics = [metrics_msi['accuracy'], metrics_msi['weighted avg']['f1-score']]
    
    # Setting up the bar positions
    bar_width = 0.25
    index = np.arange(len(metrics_names))
    
    fig, ax = plt.subplots()
    
    bar1 = plt.bar(index, orig_metrics, bar_width, label='Original Data', color='b', alpha=0.8)
    bar2 = plt.bar(index + bar_width, fused_metrics, bar_width, label='Fused Data', color='g', alpha=0.8)
    bar3 = plt.bar(index + 2 * bar_width, msi_metrics, bar_width, label='MSI Data', color='r', alpha=0.8)
    
    # Add some text for labels, title and axes ticks
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Comparison of Classification Performance')
    plt.xticks(index + bar_width, metrics_names)
    plt.legend()
    
    # Label the bars
    for bar in [bar1, bar2, bar3]:
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height, '%f' % height, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data
    original_data, fused_data, labels, labels_fused, data_msi, labels_msi = load_data()
    
    # Check if the datasets are of the same shape
    if original_data.shape != fused_data.shape or labels.shape != labels_fused.shape:
        print("Warning: The shapes of the original and fused datasets do not match.")
    if original_data.shape != data_msi.shape or labels.shape != labels_msi.shape:
        print("Warning: The shapes of the original and MSI datasets do not match.")
    
    # Reshape original, fused, and MSI data for model training
    original_data_2d = original_data.reshape((-1, original_data.shape[-1]))
    fused_data_2d = fused_data.reshape((-1, fused_data.shape[-1]))
    msi_data_2d = data_msi.reshape((-1, data_msi.shape[-1]))

    labels_2d = labels.reshape(-1)
    labels_fused_2d = labels_fused.reshape(-1)
    labels_msi_2d = labels_msi.reshape(-1)

    # Split data for the original dataset
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(original_data_2d, labels_2d, test_size=0.3, stratify=labels_2d, random_state=42)

    # Split data for the fused dataset
    X_train_fused, X_test_fused, y_train_fused, y_test_fused = train_test_split(fused_data_2d, labels_fused_2d, test_size=0.3, stratify=labels_fused_2d, random_state=42)

    # Split data for the MSI dataset
    X_train_msi, X_test_msi, y_train_msi, y_test_msi = train_test_split(msi_data_2d, labels_msi_2d, test_size=0.3, stratify=labels_msi_2d, random_state=42)

    # Train model and get metrics on original data
    y_pred_orig, metrics_orig = train_and_evaluate(X_train_orig, y_train_orig, X_test_orig, y_test_orig)
    
    # Train model and get metrics on fused data
    y_pred_fused, metrics_fused = train_and_evaluate(X_train_fused, y_train_fused, X_test_fused, y_test_fused)

    # Train model and get metrics on MSI data
    y_pred_msi, metrics_msi = train_and_evaluate(X_train_msi, y_train_msi, X_test_msi, y_test_msi)


    # Compare performance metrics
    print(f"Accuracy on original data: {metrics_orig['accuracy']}")
    print(f"Accuracy on fused data: {metrics_fused['accuracy']}")
    print(f"Accuracy on MSI data: {metrics_msi['accuracy']}")
    
    print(f"F1-Score (weighted) on original data: {metrics_orig['weighted avg']['f1-score']}")
    print(f"F1-Score (weighted) on fused data: {metrics_fused['weighted avg']['f1-score']}")
    print(f"F1-Score (weighted) on MSI data: {metrics_msi['weighted avg']['f1-score']}")


    # Plot the metrics
    plot_metrics(metrics_orig, metrics_fused, metrics_msi)

    # Visualize the classifications
    # 1. Train model on the training set for each dataset
    clf_orig = SVC(kernel='rbf')
    clf_orig.fit(X_train_orig, y_train_orig)

    clf_fused = SVC(kernel='rbf')
    clf_fused.fit(X_train_fused, y_train_fused)

    clf_msi = SVC(kernel='rbf')
    clf_msi.fit(X_train_msi, y_train_msi)
    
    # 2. Predict the full dataset (training + test)
    y_pred_orig_full = clf_orig.predict(original_data_2d)
    y_pred_fused_full = clf_fused.predict(fused_data_2d)
    y_pred_msi_full = clf_msi.predict(msi_data_2d)

    # 3. Reshape the predicted labels
    y_pred_orig_reshape = y_pred_orig_full.reshape(labels.shape)
    y_pred_fused_reshape = y_pred_fused_full.reshape(labels_fused.shape)
    y_pred_msi_reshape = y_pred_msi_full.reshape(labels_msi.shape)

    # 4. Visualize the classifications
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(y_pred_orig_reshape, cmap='nipy_spectral')
    plt.title("Predicted Labels: Original")

    plt.subplot(1, 3, 2)
    plt.imshow(y_pred_fused_reshape, cmap='nipy_spectral')
    plt.title("Predicted Labels: Fused")

    plt.subplot(1, 3, 3)
    plt.imshow(y_pred_msi_reshape, cmap='nipy_spectral')
    plt.title("Predicted Labels: MSI")

    plt.show()