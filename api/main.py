from io import BytesIO

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tensorflow.keras import models, layers


app = FastAPI()

IMAGE_SIZE = 256
n_classes = 3
input_shape = (1, IMAGE_SIZE, IMAGE_SIZE, n_classes)
class_names = ['Early Blight', 'Late Blight', 'Healthy']


resize_and_rescale_layer = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
])


def bytes2image(image_as_bytes: bytes) -> np.ndarray:
    image = np.array(Image.open(BytesIO(image_as_bytes)))
    return image


def create_and_load_model():
    model = models.Sequential([
        resize_and_rescale_layer, # data_augmentation_layer,
        layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    model.build(input_shape=input_shape)
    model.load_weights('./saved_models/baseline_99acc')
    model.trainable = False
    return model


@app.get("/ping")
async def ping() -> str:
    return "Hello, I'm alive"


@app.post("/predict")
async def predict(file: UploadFile = File()) -> dict[str, float | str]:
    image = bytes2image(await file.read())
    model = create_and_load_model()
    predictions = model.predict(np.expand_dims(image, axis=0))
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='localhost')
