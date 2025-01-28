import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data):
        """
        Initialize the EDA class with a pandas DataFrame.
        :param data: pandas DataFrame
        """
        self.data = data

    def overview(self):
        """
        Display basic information about the dataset.
        """
        print("\nDataset Overview:\n")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nColumn Data Types:\n")
        print(self.data.dtypes)
        print("\nMissing Values per Column:\n")
        print(self.data.isnull().sum())

    def summary_statistics(self):
        """
        Display summary statistics for numerical and categorical features.
        """
        print("\nSummary Statistics for Numerical Features:\n")
        print(self.data.describe())
        print("\nSummary Statistics for Categorical Features:\n")
        print(self.data.describe(include=['object', 'category']))

    def plot_numerical_distribution(self):
        """
        Plot the distribution of numerical features.
        """
        numerical_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        for column in numerical_columns:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.data[column], kde=True, bins=30, color='blue')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

    def plot_categorical_distribution(self):
        """
        Plot the distribution of categorical features.
        """
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        for column in categorical_columns:
            plt.figure(figsize=(8, 4))
            sns.countplot(data=self.data, x=column, palette='viridis')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.show()

    def correlation_analysis(self):
        """
        Display a heatmap of correlations between numerical features.
        """
        plt.figure(figsize=(10, 6))
        numerical_data = self.data.select_dtypes(include=['int64', 'float64'])
        corr_matrix = numerical_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()

    def identify_missing_values(self):
        """
        Visualize missing values in the dataset.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
        plt.title("Missing Values Heatmap")
        plt.show()

    def detect_outliers(self):
        """
        Use box plots to detect outliers in numerical features.
        """
        numerical_columns = self.data.select_dtypes(include=['int64', 'float64']).columns
        for column in numerical_columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[column], color='orange')
            plt.title(f"Outliers in {column}")
            plt.xlabel(column)
            plt.show()

