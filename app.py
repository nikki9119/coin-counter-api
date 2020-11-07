import os
from flask import Flask, jsonify, request
import base64
from detect import Detect

app = Flask(__name__)

def parsefromtext():
    one,two,five,ten=0,0,0,0
    if 'image.txt' in os.listdir('./OUTPUTS'):
        with open('OUTPUTS/image.txt','r') as fp:
            lines = fp.readlines()
            for line in lines:
                if line[0]=='0':
                    one+=1
                elif line[0]=='1':
                    ten+=1
                elif line[0]=='2':
                    two+=1
                elif line[0]=='3':
                    five+=1
    return (one,two,five,ten)

def encode_image():
    with open('OUTPUTS/image.jpg','rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        print('Encoded successfully')
        encoded_string_utf8 = encoded_string.decode("utf-8")
        return str(encoded_string_utf8)

@app.route("/")
def hello():
    return jsonify({"about":"Hello World"})

@app.route("/predict",methods=['GET','POST'])
def upload():
    print('Inside Upload')
    if 'INPUTS' not in os.listdir():
        os.mkdir('INPUTS')
        print('INPUTS folder created')
    if 'OUTPUTS' not in os.listdir():
        os.mkdir('OUTPUTS')
        print('OUTPUTS folder created')
    if request.method == 'POST':
        print('Inside Post')

        f = request.get_json()
        imgstring = f['image']
        imgdata = base64.b64decode(imgstring)
        filename = './INPUTS/image.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)
        Detect()
        one,two,five,ten = parsefromtext()
        total = (one*1)+(two*2)+(five*5)+(ten*10)
        imgstring = encode_image()
        return jsonify({
            "preds":"ok",
            "results":{
                "one":one,
                "two":two,
                "five":five,
                "ten":ten,
                "total":total
            },
            "image_en":imgstring
        })
    return 'Invalid request'

if __name__ ==  '__main__':
    app.run()