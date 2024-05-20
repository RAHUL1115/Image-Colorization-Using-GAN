# import libraries
from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import numpy as np
from base64 import b64decode, b64encode
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from uvicorn import run
import argparse

# set envs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
PYTHON_ENV = os.getenv("PYTHON_ENV")

# variables
img_size = 120
generator = load_model('src/model/gen.h5',compile=False)

# init the fast api
app = FastAPI()
app.mount("/static", StaticFiles(directory="src/static"), name="static")  # Optional for CSS, JS, etc.

templates = Jinja2Templates(directory="src/template")

# routes
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def preprocess_image(base64_str):
    """Preprocess a base64 image for the generator."""
    target_size=(img_size, img_size)
    target_array_size=(img_size, img_size)

    image_data = b64decode(base64_str.split(',')[1])

    bw_image = Image.open(BytesIO(image_data))
    bw_image = bw_image.resize(target_size).convert('L')
    bw_image = np.asarray(bw_image).reshape(target_size)
    bw_image = np.asarray( bw_image ).reshape(target_array_size)

    image_array = bw_image / 255.0
    return image_array


@app.post("/api/v1/process-image")
async def post_process_image(request: Request):
    data = await request.json()
    base64_str = data['image']

    try:
        # Process the input image
        image_array = preprocess_image(base64_str)
        generated_image = generator(np.array([image_array])).numpy()

        # Convert the output to base64
        output_image = Image.fromarray( ( generated_image[0] * 255 ).astype( 'uint8' ) ).resize( ( img_size , img_size ) )

        buffered = BytesIO()
        output_image.save(buffered, format="PNG")  # Choose an image format
        base64_output = b64encode(buffered.getvalue()).decode('utf-8')

        return {"generatedImage": f"data:image/png;base64,{base64_output}"}

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"error": "Failed to process image"} 
