"""Flask application for the web interface."""
# coding=utf-8
import sys
import os

# Flask utils
from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))
from prediction import Prediction

# Instantiate prediction class
prediction = Prediction()


# Define a flask app
app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load pickle model
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    """Render the index page."""
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def handle_upload():
    """Handle the upload of a file."""
    if request.method == 'POST':
        return prediction.handle_df_upload(request, secure_filename, app)
    elif request.method == 'GET':
        return {"status": "fail", "error": "No Get Route"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', debug=True, port=port)
