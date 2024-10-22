import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import ipaddress
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
class CREDIT_EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def missing_percentage(self):
        # Calculate the percentage of missing values
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        # Create a DataFrame to display the results nicely
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        return missing_df
    def data_types(self):
        data_typs = self.df.dtypes
        # Create a DataFrame for outlier percentages
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'data_typs': data_typs
        }).sort_values(by='data_typs', ascending=False)
        return missing_df
    def outlier_check_perc(self):
        # Filter for numeric columns
        numeric_df = self.df.select_dtypes(include=[float, int])
        # Ensure there is at least one numeric column
        if numeric_df.empty:
            raise ValueError("No numeric columns available for outlier detection.")
        # Identify outliers using IQR method
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        # Calculate outliers for each column
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        # Count total values per column and calculate percentage of outliers
        total_values = numeric_df.count()  # Non-NaN values in each column
        outlier_percentage = (outliers.sum() / total_values) * 100
        # Create a DataFrame for outlier percentages
        outlier_df = pd.DataFrame({
            'Column': numeric_df.columns,
            'outlier_percentage': outlier_percentage
        }).sort_values(by='outlier_percentage', ascending=False)
        
        return outlier_df
    def normalize_and_scale(self):
        """Normalizes and scales numerical features."""
        # Identify numerical features for scaling
        numerical_features = ['Amount','Time']
        # Initialize the StandardScaler
        scaler = StandardScaler()
        # Fit and transform the numerical features
        self.df[numerical_features] = scaler.fit_transform(self.df[numerical_features])
        print("Numerical features scaled successfully.")
        return self.df
