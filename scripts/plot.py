"""Plotting script for the results of the simulation."""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from logger import Logger
from pandas.plotting import scatter_matrix


class Plot:
    def __init__(self) -> None:
        """Initilize class."""
        try:
            self.logger = Logger("plot.log").get_app_logger()
            self.logger.info(
                'Successfully Instantiated Preprocessing Class Object')
        except Exception:
            self.logger.exception(
                'Failed to Instantiate Preprocessing Class Object')
            sys.exit(1)

    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        """Plot the hist of the column.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            column(str): column to be plotted.
            color(str): color of the histogram.
        """
        # plt.figure(figsize=(15, 10))
        # fig, ax = plt.subplots(1, figsize=(12, 7))
        sns.displot(data=df, x=column, color=color,
                    kde=True, height=7, aspect=2)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        self.logger.info(
            'Plotting a histogram')
        plt.show()

    def plot_count(self, df: pd.DataFrame, column: str) -> None:
        """Plot the count of the column.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            column(str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        self.logger.info(
            'Plotting a plot_count')
        sns.countplot(df, hue=column)
        plt.title(f'Distribution of {column}', size=20, fontweight='bold')
        plt.show()

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
        """Plot bar of the column.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            x_col(str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        self.logger.info(
            'Plotting a bar chart')
        plt.show()

    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        """Plot Heat map of the dataset.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            title(str): title of chart.
        """
        # num_cols = df.select_dtypes(include=np.number).columns
        plt.figure(figsize=(12, 7))
        sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                    vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
        plt.title(title, size=18, fontweight='bold')
        self.logger.info(
            'Plotting a heatmap for the dataset: ')
        plt.show()

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        """Plot box chart of the column.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            x_col(str): column to be plotted.
            title(str): title of chart.
        """
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        self.logger.info(
            'Plotting a box plot for Column: ', x_col)
        plt.show()

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        """Plot the box chart for multiple column.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            column(str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x=x_col, y=y_col)
        plt.title(title, size=20)
        plt.xticks(rotation=75, fontsize=14)
        plt.yticks(fontsize=14)
        self.logger.info(
            'Plotting a multiple box plot: ')
        plt.show()

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
        """Plot Scatter chart of the data.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            column(str): column to be plotted.
        """
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
        plt.title(title, size=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        self.logger.info(
            'Plotting a scatter plot')
        plt.show()

    def plot_pie(self, data, labels, title) -> None:
        """Plot pie chart of the data.

        Args:
            data(list): Data to be plotted.
            labels(list): labels of the data.
            colors(list): colors of the data.
        """
        plt.figure(figsize=(12, 7))
        colors = sns.color_palette('bright')
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        plt.title(title, size=20)
        self.logger.info(
            'Plotting a pie chart')
        plt.show()

    # function to get the values in a plot

    def get_value(self, figure):
        """Get values in a plot.

        Args:
            figure(_type_): _description_
        """
        self.logger.info(
            'Getting value for a plot')
        for p in figure.patches:
            figure.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2.,
                                                     p.get_height()), ha='center', va='center',
                            xytext=(0, 10), textcoords='offset points')

    # function to set figure parameters

    def fig_att(self, figure, title, titlex, titley, size, sizexy, weight):
        """Plot chart of the data.

        Args:
            figure(_type_): figure to be plotted.
            title(_type_): title of plot
            titlex(_type_): x axis title
            titley(_type_): y axis title
            size(_type_): size of plot
            sizexy(_type_): size of x and y axis
            weight(_type_): weight of title
        """
        # setting the parameters for the title, x and y labels of the plot
        figure.set_title(title, size=size, weight=weight)
        figure.set_xlabel(titlex, size=sizexy, weight=weight)
        figure.set_ylabel(titley, size=sizexy, weight=weight)
        self.logger.info(
            'set figure parameters')

    # function to change rotation of the x axis tick labels
    def rotate(self, figure, rotation):
        """Rotate the x axis tick labels.

        Args:
            figure(_type_): figure to be plotted.
            rotation(_type_): rotation of x axis tick labels
        """
        # changing the rotation of the x axis tick labels
        self.logger.info(
            'Plotting a chart')
        for item in figure.get_xticklabels():
            item.set_rotation(rotation)

    def sc_matrix(self, df: pd.DataFrame, title: str) -> None:
        """Plot the scatter matrix of the data.

        Args:
            df(pd.DataFrame): Dataframe to be plotted.
            title(str): title of chart.
        """
        plt.figure(figsize=(12, 7))
        sns.pairplot(df)
        plt.title(title, size=20)
        self.logger.info(
            'Plotting a scatter matrix')
        scatter_matrix(df, alpha=0.2, figsize=(12, 7), diagonal='kde')
        # plt.show()
