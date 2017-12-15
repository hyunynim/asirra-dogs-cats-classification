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
    :return: X_set, y_set
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
        y_set = y_set_oh
    print('\nDone')

    return X_set, y_set


def random_crop_reflect(images, crop_l):
    """
    Perform random cropping and reflecting of images.
    :param images: np.ndarray, shape: (n, H, W, C)
    :param crop_l: Integer, a side length of crop region.
    :return:
    """
    H, W = images.shape[1:3]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        # Randomly crop image
        y = np.random.randint(H-crop_l)
        x = np.random.randint(W-crop_l)
        image = image[y:y+crop_l, x:x+crop_l]

        # Randomly reflect image, horizontally
        reflect = bool(np.random.randint(2))
        if reflect:
            image = image[:, ::-1]

        augmented_images.append(image)
    return np.stack(augmented_images)    # shape: (n, H, W, C)


class DataSet(object):
    def __init__(self, images, labels, seed=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C).
        :param labels: np.ndarray, shape: (N,) or (N, num_classes).
        :param seed:
        """
        assert images.shape[0] == labels.shape[0], (
            'Number of examples mismatch, between images and labels.'
        )
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True, augment=True):
        """
        Return the next `batch_size` examples from this dataset.
        :param batch_size: Integer, size of a single batch.
        :param shuffle: Boolean, whether to shuffle the whole set while sampling batch.
        :return:
        """
        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self._images[perm0]
            self._labels = self._labels[perm0]

        # Go to the next epoch, if current index goes beyond the total number of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index
            images_rest_part = self._images[start_index:self._num_examples]
            labels_rest_part = self._labels[start_index:self._num_examples]

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self._images[perm]
                self._labels = self._labels[perm]

            # Start the next epoch
            start_index = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end_index = self._index_in_epoch
            images_new_part = self._images[start_index:end_index]
            labels_new_part = self._labels[start_index:end_index]

            batch_images = np.concatenate((images_rest_part, images_new_part), axis=0)
            batch_labels = np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end_index = self._index_in_epoch
            batch_images = self._images[start_index:end_index]
            batch_labels = self._labels[start_index:end_index]

        if augment:
            batch_images = random_crop_reflect(batch_images, 227)

        return batch_images, batch_labels

