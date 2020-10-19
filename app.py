import keras_preprocessing
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import load_model, model_from_json

import os
import numpy as np
import base64

from flask import Flask, jsonify, request
# from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('./rps.h5')
print('Model loaded')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(300, 300))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    preds = model.predict(images,batch_size=10)
    return preds

@app.route("/")
def hello():
    return jsonify({"about":"Hello World"})

@app.route("/get_name")
def helloname():
    return jsonify({"about":"Hello World, I am Nikhil"})

@app.route("/json_req",methods=['GET','POST'])
def resendreq():
    req_str = request.get_json()
    strng = req_str['str']
    return jsonify({'str':strng})

@app.route("/predict",methods=['GET','POST'])
def upload():
    print('Inside Upload')
    if request.method == 'POST':
        print('Inside Post')

        f = request.get_json()
        imgstring = f['image']
        imgdata = base64.b64decode(imgstring)
        filename = 'some_image.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)

        preds = model_predict(filename, model)
        print(preds)
        max = 0.0
        if(preds[0][0] and preds[0][0] > max):
            max = preds[0][0]
            value = 1
        if(preds[0][1] and preds[0][1] > max):
            max = preds[0][1]
            value = 10
        if(preds[0][2] and preds[0][2] > max):
            max = preds[0][2]
            value = 2
        if(preds[0][3] and preds[0][3] > max):
            max = preds[0][3]
            value = 5
        return jsonify({"preds":value})
    return None


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')