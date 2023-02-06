from math import floor

import tensorflow as tf
from tensorflow.keras import models, layers
from typing import Tuple


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50


def train_val_test_split(dataset: tf.data.Dataset,
                         train_split: float = 0.8,
                         val_split: float = 0.1,
                         shuffle: bool = True,
                         shuffle_size: int = 10000,
                         random_seed: int = 12
                         ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Makes a train, validation and test separation of a given tf.data.Dataset
    Args:
        dataset:
        train_split:
        val_split:
        shuffle:
        shuffle_size:
        random_seed:

    Returns:

    """
    assert train_split + val_split < 1, "Test part can't be empty!"
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed=random_seed)

    train_size = floor(train_split * len(dataset))
    val_size = floor(val_split * len(dataset))

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.take(val_size)
    test_dataset = test_dataset.skip(val_size)

    return train_dataset, val_dataset, test_dataset


resize_and_rescale_layer = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation_layer = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])
