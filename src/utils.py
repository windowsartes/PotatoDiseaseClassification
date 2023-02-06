from io import BytesIO
from math import floor

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import models, layers
from typing import Tuple


IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
N_CLASSES = 3

resize_and_rescale_layer = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])

data_augmentation_layer = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    layers.experimental.preprocessing.RandomRotation(0.2)
])


def bytes2image(image_as_bytes: bytes) -> np.ndarray:
    """
    Decode a sequence of bytes into np.array.
    @param image_as_bytes: sequence of bytes. For example, from UploadFile().read()
    @return: np.array with decoded byte sequence.
    """
    image = np.array(Image.open(BytesIO(image_as_bytes)))
    return image


def train_val_test_split(dataset: tf.data.Dataset,
                         train_split: float = 0.8,
                         val_split: float = 0.1,
                         shuffle: bool = True,
                         shuffle_size: int = 10000,
                         random_seed: int = 12
                         ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Makes a train-val-test split with given tf.data.Dataset instance.
    @param dataset: tf.data.Dataset instance you want to split.
    @param train_split: proportion of the training part.
    @param val_split: proportion of the validation part.
    @param shuffle: should data be shuffled or not.
    @param shuffle_size: shuffle buffer size. See tf.data.Dataset's shuffle method.
    @param random_seed: shuffle random seed.
    @return: dataset, divided into train, val and test parts.
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


def create_and_load_model(path: str, input_shape: tuple[int, int, int, int]) -> tf.keras.Model:
    """
    Creates and model with baseline architecture and loads pre-trained weights.
    @param path: path to saved with model.save_weights() model's weights.
    @param input_shape: input shape for model.build().
    @return: model with pre-trained weights.
    """
    model = models.Sequential([
        resize_and_rescale_layer,  # data_augmentation_layer,
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(N_CLASSES, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    model.build(input_shape=input_shape)
    model.load_weights(path)
    model.trainable = False

    return model
