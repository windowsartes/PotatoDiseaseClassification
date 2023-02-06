import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile

from src.utils import IMAGE_SIZE, CHANNELS, bytes2image, create_and_load_model


INPUT_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

app = FastAPI()


@app.get("/ping")
async def ping() -> str:
    return "Hello, I'm alive"


@app.post("/predict")
async def predict(file: UploadFile = File()) -> dict[str, float | str]:
    image = bytes2image(await file.read())
    model = create_and_load_model('./saved_models/baseline_99acc', input_shape=INPUT_SHAPE)
    predictions = model.predict(np.expand_dims(image, axis=0))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='localhost')
