"""Test the preprocess module."""
import os
import sys
import unittest

import numpy
import pandas as pd
# from clean_tweets_dataframe import Clean_Tweets
from pandas._libs.tslibs.timestamps import Timestamp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from preprocess import Preprocess
from logger import Logger


class TESTPHARMASALES(unittest.TestCase):
    """A class for unit-testing function in the preprocess.py file.

    Args:
        unittest.TestCase this allows the new class to inherit
        from the unittest module
    """

    def setUp(self) -> pd.DataFrame:
        """Dataframe that contains the data.

        Returns:
            pd.DataFrame: DF from train_joined.csv file.
        """
        self.df = self.df = pd.DataFrame({'Date':
                                         '4/4/2021 12:01', 'Store': 5, 'DayOfWeek': 3, 'Sales': 232, 'Customers': 4321, 'Open': 1, 'SchoolHoliday': 1, 'StateHoliday': 1, 'StoreType': 'A', 'Assortment': 'c', 'CompetitionDistance': 133, 'CompetitionOpenSinceMonth': 1999, 'Year': 1999, 'weekOfyear': 12}, {'Date':
                                         '8/5/2014 12:01', 'Store': 4, 'DayOfWeek': 1, 'Sales': 572, 'Customers': 1321, 'Open': 0, 'SchoolHoliday': 0, 'StateHoliday': 1, 'StoreType': 'D', 'Assortment': 'b', 'CompetitionDistance': 432, 'CompetitionOpenSinceMonth': 2000, 'Year': 2001, 'weekOfyear': 21},
                                         {'Date':
                                         '8/5/2014 12:01', 'Store': 4, 'DayOfWeek': 1, 'Sales': 572, 'Customers': 1321, 'Open': 0, 'SchoolHoliday': 0, 'StateHoliday': 1, 'StoreType': 'D', 'Assortment': 'b', 'CompetitionDistance': 432, 'CompetitionOpenSinceMonth': 2000, 'Year': 2001, 'weekOfyear': 21})

        # tweet_df = self.df.get_tweet_df()

    def test_convert_to_datetime(self):
        """Test convert to datetime module."""
        df = Preprocess().convert_to_datetime(self.df, 'Date')
        assert type(df['Date'][0]) is Timestamp
        
    def test_get_numerical_columns(self):
        """Test get numerical columns module."""
        df = Preprocess().get_numerical_columns(self.df)
        assert df == ['Store', 'DayOfWeek', 'Sales', 'Customers', 'Open', 'SchoolHoliday', 'StateHoliday', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'Year', 'weekOfyear']
        
    def test_get_categorical_columns(self):
        """Test get categorical columns module."""
        df = Preprocess().get_categorical_columns(self.df)
        assert df == ['Date', 'StoreType', 'Assortment']

    def test_drop_duplicate(self):
        """Test convert to datetime module."""
        df = Preprocess().drop_duplicate(self.df)
        assert df.shape[0] == 1


if __name__ == '__main__':
    unittest.main()
