"""Load data from a file and returns a dataframe.

Returns:
    pd.DataFrame: a dataframe with the data from the file.
"""
import json
import sys
from git import Object
import pandas as pd
from logger import Logger


class LoadData:
    """
    Load data from a file and returns a dataframe.

    Return
    ------
    dataframe
    """

    def __init__(self) -> None:
        """Initilize class."""
        try:
            self.logger = Logger("logger.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated load_data Class Object')
        except Exception:
            self.logger.exception(
                'Failed to Instantiate load_data Class Object')
            sys.exit(1)

    def read_json(self, json_file: str, dfExtractFunc: Object) -> pd.DataFrame:
        """Json file reader to open and read json files into a panda dataframe.
        Args:
        -----
        json_file: str - path of a json file
        dfExtractFunc: object - The function that will used to extract data from the list formed from the json file

        Returns
        -------
        dataframe containing data extracted from the json file
        """

        data_list = []
        for item in open(json_file, 'r'):
            data_list.append(json.loads(item))

        df = dfExtractFunc(data_list)
        return df

    def read_excel(self, excel_file, startRow=0) -> pd.DataFrame:
        """Excel file reader to open and read excel files into a panda dataframe.
        Args:
        -----
        excel_file: str - path of a excel file
        engine: str - sets the default engine used by pandas to load the excel file
        startRow: int - sets the first row in excel sheet where pandas should start loading from

        Returns
        -------
        dataframe containing data extracted from the excel file
        """
        df = pd.read_excel(excel_file, engine='openpyxl')
        return df

    def read_csv(self, csv_file) -> pd.DataFrame:
        """Csv file reader to open and read csv files into a panda dataframe.
        Args:
        -----
        csv_file: str - path of a json file

        Returns
        -------
        dataframe containing data extracted from the csv file"""
        return pd.read_csv(csv_file)
