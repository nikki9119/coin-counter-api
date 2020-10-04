import keras_preprocessing
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import load_model
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import os
# from zipfile import ZipFile
import numpy as np

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

# MODEL_PATH = 'rps.h5'

model = load_model('rps.h5')
# model._make_predict_function()

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(300, 300))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(images,batch_size=10)
    return preds

@app.route("/")
def hello():
    return jsonify({"about":"Hello World"})

# @app.route("/get_name")
# def helloname():
#     return jsonify({"about":"Hello World, I am Nikhil"})

@app.route("/predict",methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)