"""Predict on data from user."""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Prediction:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
        except Exception:
            sys.exit(1)

    def allowed_file(self, filename):
        """Allowed file."""
        ALLOWED_EXTENSIONS = {'csv'}
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def preprocess(self, path):
        """Preprocess data."""
        try:
            data = pd.read_csv(path, parse_dates=True, index_col="Date")
            data['Year'] = data.index.year
            data['Month'] = data.index.month
            data['Day'] = data.index.day
            data['WeekOfYear'] = data.index.weekofyear
            return data
        except Exception:
            self.logger.exception(
                'Failed to get Numerical Columns from Dataframe')
            sys.exit(1)

    def handle_df_upload(self, request, secure_filename, app):
        """Handle upload."""
        if request.method == 'POST':
            if 'file' not in request.files:
                # flash('No file part')
                return {"status": "fail", "error": "No file part"}
            file = request.files['file']
            if file.filename == '':
                return {"status": "fail", "error": "No file part"}
            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                print(filename)
                file_name = 'pred.csv'
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
                full_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], file_name)
                data = self.preprocess(full_path)
                print(data)
                try:
                    results = self.predict(data)
                    dates = data.index.values
                    dates = dates.astype(str).tolist()
                    sales = list(results)
                    data = {
                        "Date": dates,
                        "Sales": sales
                    }
                    return data
                except Exception:
                    return {"status": "fail", "error": "Failed to predict"}
        elif request.method == 'GET':
            return {"status": "fail", "error": "Get request not available yet"}

    def predict(self, df):
        """Predict using pickle model."""
        cols = ['StateHoliday', 'Store', 'DayOfWeek', 'Open', 'Promo',
                'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear']
        loaded_model = pickle.load(
            open("./models/2022-05-27-11-08-03.pkl", 'rb'))
        # loaded_model = pickle.load(open("./models/2022-05-26-10-09-06.pkl", 'rb'))
        df = df[cols]
        result = loaded_model.predict(df)
        result = np.exp(result)
        date = df.index.values

        new_df = pd.DataFrame()
        new_df['Date'] = date
        new_df['Predicted Sales'] = result
        print("RESULT:", result)
        # return {"date": date, "sales": result}
        return result
