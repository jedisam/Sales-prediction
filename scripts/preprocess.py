"""Preprocess the data."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from logger import Logger
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder


class Preprocess:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
            # self.logger = Logger("preprocess.log").get_app_logger()
            # self.logger.info(
            #     'Successfully Instantiated preprocess Class Object')
        except Exception:
            # self.logger.exception(
            #     'Failed to Instantiate Preprocessing Class Object')
            sys.exit(1)

    def get_numerical_columns(self, df):
        """Get numerical columns from dataframe."""
        try:
            # self.logger.info('Getting Numerical Columns from Dataframe')
            num_col = df.select_dtypes(
                exclude="object").columns.tolist()
            # num_col.remove('date')
            return num_col
        except Exception:
            # self.logger.exception(
            #     'Failed to get Numerical Columns from Dataframe')
            sys.exit(1)

    def get_categorical_columns(self, df):
        """Get categorical columns from dataframe."""
        try:
            # self.logger.info('Getting Categorical Columns from Dataframe')
            return df.select_dtypes(
                include="object").columns.tolist()
        except Exception:
            # self.logger.exception(
            #     'Failed to get Categorical Columns from Dataframe')
            sys.exit(1)

    def drop_duplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate rows."""
        # self.logger.info('Dropping duplicate row')
        df = df.drop_duplicates(subset='Date')

        # self.convert_to_datetime(self.df)
        return df

    def get_missing_values(self, df):
        """Get missing values from dataframe."""
        try:
            # self.logger.info('Getting Missing Values from Dataframe')
            return df.isnull().sum()
        except Exception:
            # self.logger.exception(
            #     'Failed to get Missing Values from Dataframe')
            sys.exit(1)

    def convert_to_datetime(self, df, column):
        """Convert column to datetime."""
        try:
            # self.logger.info('Converting Column to Datetime')
            df[column] = pd.to_datetime(df[column])
            return df
        except Exception:
            # self.logger.exception(
            #     'Failed to convert Column to Datetime')
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
            # self.logger.info('Joining two Dataframes')
            return pd.merge(df1, df2, on=on)
        except Exception:
            # self.logger.exception(
            #     'Failed to join two Dataframes')
            sys.exit(1)

    # check if it's weekend
    def is_weekend(self, date):
        """Check if it's weekend."""
        try:
            # self.logger.info('Checking if it\'s weekend')
            return 1 if (date.weekday() > 4 or date.weekday() < 1) else 0
        except Exception:
            # self.logger.exception(
            #     'Failed to Check if it\'s weekend')
            sys.exit(1)

    def extract_fields_date(self, df, date_column):
        """Extract fields from date column."""
        try:
            # self.logger.info('Extracting Fields from Date Column')
            df['Year'] = df[date_column].dt.year
            df['Month'] = df[date_column].dt.month
            df['Day'] = df[date_column].dt.day
            df['DayOfWeek'] = df[date_column].dt.dayofweek
            df['weekday'] = df[date_column].dt.weekday
            df['weekofyear'] = df[date_column].dt.weekofyear
            df['weekend'] = df[date_column].apply(self.is_weekend)
            return df
        except Exception:
            # self.logger.exception(
            #     'Failed to Extract Fields from Date Column')
            sys.exit(1)

    def get_missing_data_percentage(self, df):
        """Get missing data percentage."""
        try:
            # self.logger.info('Getting Missing Data Percentage')
            total = df.isnull().sum().sort_values(ascending=False)
            percent_1 = total/df.isnull().count()*100
            percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
            missing_data = pd.concat(
                [total, percent_2], axis=1, keys=['Total', '%'])
            return missing_data
        except Exception:
            # self.logger.exception(
            #     'Failed to Get Missing Data Percentage')
            sys.exit(1)

    def fill_missing_median(self, df, columns):
        """Fill missing data with median."""
        try:
            # self.logger.info('Filling Missing Data with Median')
            for col in columns:
                df[col] = df[col].fillna(df[col].median())
            return df
        except Exception:
            # self.logger.exception(
            #     'Failed to Fill Missing Data with Median')
            sys.exit(1)

    def fill_missing_with_zero(self, df, columns):
        """Fill missing data with zero."""
        try:
            # self.logger.info('Filling Missing Data with Zero')
            for col in columns:
                df[col] = df[col].fillna(0)
            return df
        except Exception:
            # self.logger.exception(
            #     'Failed to Fill Missing Data with Zero')
            sys.exit(1)

    def fill_missing_mode(self, df, columns):
        """Fill missing data with mode."""
        try:
            # self.logger.info('Filling Missing Data with Mode')
            for col in columns:
                df[col] = df[col].fillna(df[col].mode()[0])
            return df
        except Exception:
            # self.logger.exception(
                # 'Failed to Fill Missing Data with Mode')
            sys.exit(1)

    def replace_outliers_iqr(self, df, columns):
        """Replace outlier data with IQR."""
        try:
            # self.logger.info('Replacing Outlier Data with IQR')
            for col in columns:
                Q1, Q3 = df[col].quantile(
                    0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                cut_off = IQR * 1.5
                lower, upper = Q1 - cut_off, Q3 + cut_off

                df[col] = np.where(
                    df[col] > upper, upper, df[col])
                df[col] = np.where(
                    df[col] < lower, lower, df[col])
            return df
        except Exception:
            # self.logger.exception(
            #     'Failed to Replace Outlier Data with IQR')
            sys.exit(1)
