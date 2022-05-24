"""Preprocess the data."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from logger import Logger
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder


class Preprocess:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
            self.logger = Logger("preprocess.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated preprocess Class Object')
        except Exception:
            self.logger.exception(
                'Failed to Instantiate Preprocessing Class Object')
            sys.exit(1)

    def get_numerical_columns(self, df):
        """Get numerical columns from dataframe."""
        try:
            self.logger.info('Getting Numerical Columns from Dataframe')
            num_col = df.select_dtypes(
                exclude="object").columns.tolist()
            num_col.remove('date')
            return num_col
        except Exception:
            self.logger.exception(
                'Failed to get Numerical Columns from Dataframe')
            sys.exit(1)

    def get_categorical_columns(self, df):
        """Get categorical columns from dataframe."""
        try:
            self.logger.info('Getting Categorical Columns from Dataframe')
            return df.select_dtypes(
                include="object").columns.tolist()
        except Exception:
            self.logger.exception(
                'Failed to get Categorical Columns from Dataframe')
            sys.exit(1)

    def get_missing_values(self, df):
        """Get missing values from dataframe."""
        try:
            self.logger.info('Getting Missing Values from Dataframe')
            return df.isnull().sum()
        except Exception:
            self.logger.exception(
                'Failed to get Missing Values from Dataframe')
            sys.exit(1)

    def convert_to_datetime(self, df, column):
        """Convert column to datetime."""
        try:
            self.logger.info('Converting Column to Datetime')
            df[column] = pd.to_datetime(df[column])
            return df
        except Exception:
            self.logger.exception(
                'Failed to convert Column to Datetime')
            sys.exit(1)

    def label_encode(self, df, columns):
        """Label encode the target variable.

        Parameters
        ----------
        df: Pandas Dataframe
            This is the dataframe containing the features and target variable.
        columns: list
        Returns
        -------
        The function returns a dataframe with the target variable encoded.
        """
        # Label Encoding

        label_encoded_columns = []
        # For loop for each columns
        for col in columns:
            # We define new label encoder to each new column
            le = LabelEncoder()
            # Encode our data and create new Dataframe of it,
            # notice that we gave column name in "columns" arguments
            column_dataframe = pd.DataFrame(
                le.fit_transform(df[col]), columns=[col])
            # and add new DataFrame to "label_encoded_columns" list
            label_encoded_columns.append(column_dataframe)

        # Merge all data frames
        label_encoded_columns = pd.concat(label_encoded_columns, axis=1)
        return label_encoded_columns

    def join_dataframes(self, df1, df2, on, how="inner"):
        """Join two dataframes."""
        try:
            self.logger.info('Joining two Dataframes')
            return pd.merge(df1, df2, on=on)
        except Exception:
            self.logger.exception(
                'Failed to join two Dataframes')
            sys.exit(1)

    def extract_fields_date(self, df, date_column):
        """Extract fields from date column."""
        try:
            self.logger.info('Extracting Fields from Date Column')
            df['Year'] = df[date_column].dt.year
            df['Month'] = df[date_column].dt.month
            df['Day'] = df[date_column].dt.day
            df['DayOfWeek'] = df[date_column].dt.dayofweek

            return df
        except Exception:
            self.logger.exception(
                'Failed to Extract Fields from Date Column')
            sys.exit(1)
