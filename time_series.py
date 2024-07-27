from motion_code.motion_code import MotionCode
from motion_code.data_processing import load_data, process_data_for_motion_codes
import numpy as np
from motion_code.sparse_gp import sigmoid


def get_most_informative_img_sri(img_sri, gt, num_classes,
                                 sample_size=25,
                                 num_most_informative_channels=8):
    # Turn original images into collection of timeseries, each corresponds to a single pixel.
    img_hsi_reshaped = img_sri.reshape(-1, img_sri.shape[-1])
    width, height = img_sri.shape[0], img_sri.shape[1]
    gt_reshaped = gt.reshape(-1, gt.shape[-1])
    indices = None
    all_labels = np.argmax(gt_reshaped, axis=1)
    for c in range(num_classes):
        indices_in_class = np.where(all_labels == c)[0]
        current_choices = np.random.choice(indices_in_class, size=sample_size)
        if indices is None:
            indices = current_choices
        else:
            indices = np.append(indices, current_choices)
    num_series = indices.shape[0]
    num_channels = img_hsi_reshaped.shape[1]
    Y_train = img_hsi_reshaped[indices, :].reshape(num_series, 1, -1)
    labels_train = np.argmax(gt_reshaped[indices, :], axis=1)

    # Motion code train to extract most informative timestamps
    X_train, Y_train, labels_train = process_data_for_motion_codes(Y_train, labels_train)
    model = MotionCode(m=num_most_informative_channels, Q=1, latent_dim=2, sigma_y=0.1)
    print('Training motion code...')
    model_path = 'motion_code/saved_models/' + 'test_model'
    model.fit(X_train, Y_train, labels_train, model_path)
    model_path = 'motion_code/saved_models/test_model'
    model.load(model_path)
    print('Done motion code training')

    # From most informative timestamps, get best extracted HSI.
    num_motion = np.unique(labels_train).shape[0]
    X_m, Z = model.X_m, model.Z
    revised_hsi = np.zeros((img_hsi_reshaped.shape[0], num_most_informative_channels))
    for k in range(num_motion):
        most_informative_indices = np.rint(num_channels * sigmoid(X_m @ Z[k])).astype(int)
        revised_hsi[all_labels==k, :] = img_hsi_reshaped[all_labels==k][:, most_informative_indices]

    return revised_hsi.reshape(width, height, -1)