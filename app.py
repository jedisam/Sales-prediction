from __future__ import division, print_function
# coding=utf-8
import sys
import os
import shutil
import glob
import re
import numpy as np
import tensorflow as tf
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def delete_file():
    if os.path.exists("demofile.txt"):
        os.remove("demofile.txt")
    else:
        print("The file does not exist")


def model_predict(img_path, model):
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255.)
    
    test_generator = test_datagen.flow_from_directory("./uploads",
                                                      batch_size=4,
                                                      class_mode=None,
                                                      target_size=(100, 100))
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')
    test_generator.reset()
    preds = model.predict(test_generator)
    print("preds", preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global class_names
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/zz', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        shutil.rmtree('./uploads/zz')
        os.mkdir('./uploads/zz')

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        predicted_class_indices = np.argmax(preds, axis=1)
        # print("predicted indices yididiyaaa mikiii des yebelachuh: betam yikirta///des beluachewal", predicted_class_indices )
        res = class_names[predicted_class_indices[0]]
        print('resss====', res)
        # result = str(pred_class[0][0][1])
        return res
    return None


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.run(host='0.0.0.0', debug=True, port=port)
