from flask import Flask, render_template, request

import numpy as np
import os
import urllib.request
from PIL import Image
from tensorflow.python.keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

img_size = 120
generator = load_model('model/gen.h5',compile=False)

def load_input(imgpath):
    bwimage = Image.open(imgpath).resize(( img_size , img_size ))
    bwimage = bwimage.convert( 'L' )
    bwimage = ( np.asarray( bwimage ).reshape( ( img_size , img_size , 1 ) ) ) / 255
    return bwimage
	
def load_input_url(imgpath):
    urllib.request.urlretrieve(imgpath,"static/input.jpg")
    bwimage = Image.open("static/input.jpg").resize(( img_size , img_size ))
    bwimage = bwimage.convert( 'L' )
    bwimage = ( np.asarray( bwimage ).reshape( ( img_size , img_size , 1 ) ) ) / 255
    return bwimage


@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/input.png')

        img = load_input('static/input.png')
        output_array = generator(np.array([img])).numpy()
        output = Image.fromarray( ( output_array[0] * 255 ).astype( 'uint8' ) ).resize( ( img_size , img_size ) )
        output.save("static/ouput.png")

        return render_template('home.html',uploaded=True,)
    else:
        return render_template('home.html',uploaded=False)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == '__main__':
    app.run(debug = True)