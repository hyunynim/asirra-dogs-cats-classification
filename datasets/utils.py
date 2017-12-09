import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize


def read_asirra_subset(subset_dir, one_hot=True):
    """
    Load the Asirra Dogs vs. Cats data subset from disk
    and perform preprocessing to prepare it for classifiers.
    :param root_dir: String, giving path to the directory to read.
    :param one_hot: Boolean, whether to return one-hot encoded labels.
    :return:
    """
    # Read trainval data
    X_set, y_set = [], []
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)
    for i, filename in enumerate(filename_list):
        if i % 1000 == 0:
            print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')
        label = filename.split('.')[0]
        if label == 'cat':
            y = 0
        else:  # label == 'dog'
            y = 1
        file_path = os.path.join(subset_dir, filename)
        img = imread(file_path)
        # shape: (H, W, 3), range: [0, 255]
        img_resized = resize(img, (256, 256), mode='constant').astype(np.float32)
        # shape: (256, 256, 3), range: [0.0, 1.0]
        X_set.append(img_resized)
        y_set.append(y)

    # Stack all subset data
    X_set = np.stack(X_set)  # shape: (N, 256, 256, 3)
    y_set = np.stack(y_set).astype(np.uint8)  # shape: (N,)

    if one_hot:
        # Convert labels to one-hot vectors
        y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
        y_set_oh[np.arange(set_size), y_set] = 1
        y_trainval = y_set_oh
    print('\nDone')

    return X_set, y_set
