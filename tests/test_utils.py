from io import BytesIO

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.utils import train_val_test_split, IMAGE_SIZE, BATCH_SIZE
from src.utils import bytes2image


test_cases = []

real_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                    "PlantVillage",
                    shuffle=True,
                    image_size=(IMAGE_SIZE, IMAGE_SIZE),
                    batch_size=BATCH_SIZE
               )
test_cases.append((real_dataset, 54, 6, 8))

example_dataset_1 = tf.keras.utils.timeseries_dataset_from_array(
                    np.arange(100), None, sequence_length=10, sequence_stride=3, sampling_rate=3, batch_size=2)
test_cases.append((example_dataset_1, 9, 1, 2))

example_dataset_2 = tf.keras.utils.timeseries_dataset_from_array(
                    np.arange(1000), None, sequence_length=50, sequence_stride=8, sampling_rate=5, batch_size=2)
test_cases.append((example_dataset_2, 37, 4, 6))

example_dataset_3 = tf.keras.utils.timeseries_dataset_from_array(
                    np.arange(9), None, sequence_length=1, sequence_stride=1, sampling_rate=1, batch_size=1)
test_cases.append((example_dataset_3, 7, 0, 2))


@pytest.mark.parametrize('dataset, len_train, len_val, len_test', test_cases)
def test_train_val_split_function(dataset: tf.data.Dataset, len_train: int, len_val: int, len_test: int) -> None:
    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset)
    assert len(train_dataset) == len_train
    assert len(val_dataset) == len_val
    assert len(test_dataset) == len_test


test_cases = ['tests/test_image_dump/1.JPG',
              'tests/test_image_dump/2.JPG',
              'tests/test_image_dump/3.JPG',
              'tests/test_image_dump/4.JPG',
              'tests/test_image_dump/5.JPG',
              'tests/test_image_dump/6.JPG',
              'tests/test_image_dump/7.JPG',
              'tests/test_image_dump/8.JPG',
              'tests/test_image_dump/9.JPG',
              'tests/test_image_dump/10.JPG',
              'tests/test_image_dump/11.JPG',
              'tests/test_image_dump/12.JPG']


@pytest.mark.parametrize('path_to_image', test_cases)
def test_bytes2image(path_to_image: str) -> None:
    """
    Tests bytest2image.
    Due to this time unknown random, we allow maximum loss in pixel equal to 20 points.
    @param path_to_image: path to test image we want to encode and decode then.
    @return:
    """
    with Image.open(path_to_image) as image:
        buff = BytesIO()
        image_as_np_array = np.array(image)
        image.save(buff, format='JPEG')
        image_as_bytes = buff.getvalue()
        image_decoded = bytes2image(image_as_bytes)
        height, width, _ = image_as_np_array.shape
        for i in range(height):
            for j in range(width):
                first_image_pixel = image_decoded[i][j].astype('int64')
                second_image_pixel = image_as_np_array[i][j].astype('int64')
                assert np.max(np.abs(first_image_pixel - second_image_pixel)) < 20
