import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import ipaddress
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders as ce
class FRAUD_EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.merged_data = None
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

    def change_col_date_time(self, cols, date_format=None):
        for col in cols:
            try:
                if date_format:
                    # Convert with a specific format
                    self.df[col] = pd.to_datetime(self.df[col], format=date_format, errors='coerce')
                else:
                    # Auto-detect format
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"Successfully converted column '{col}' to datetime.")
            except Exception as e:
                print(f"Error converting column '{col}' to datetime: {e}")
        return self.df

    def histogram_boxplot(self, feature, figsize=(12, 7), kde=False, bins=None):
        """
        Boxplot and histogram combined
        kde: whether to show the density curve (default False)
        bins: number of bins for histogram (default None)
        """
        f2, (ax_box2, ax_hist2) = plt.subplots(
            nrows=2,  # Number of rows of the subplot grid = 2
            sharex=True,  # x-axis will be shared among all subplots
            gridspec_kw={"height_ratios": (0.25, 0.75)},
            figsize=figsize,
        )  # creating the 2 subplots

        # Boxplot
        sns.boxplot(
            data=self.df, x=feature, ax=ax_box2, showmeans=True, color="violet"
        )
        ax_box2.set_title(f"Boxplot of {feature}")  # Add title for boxplot

        # Histogram
        if bins:
            sns.histplot(data=self.df, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")
        else:
            sns.histplot(data=self.df, x=feature, kde=kde, ax=ax_hist2)

        ax_hist2.axvline(self.df[feature].mean(), color="green", linestyle="--", label=f"Mean: {self.df[feature].mean():.2f}")
        ax_hist2.axvline(self.df[feature].median(), color="black", linestyle="-", label=f"Median: {self.df[feature].median():.2f}")
        ax_hist2.legend()  # Add legend for mean and median lines
        ax_hist2.set_title(f"Histogram of {feature}")  # Add title for histogram

        # Add a super title for the entire figure
        f2.suptitle(f"Boxplot and Histogram for {feature}", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit suptitle
        plt.show()

    def bar_plot(self, features, hue=None, figsize=(12, 7)):
        """
        Bar plots for multiple categorical features using subplots with count labels.
        :param features: List of categorical features for which to plot bar charts.
        :param hue: Optional categorical variable that will determine the color of the bars.
        :param figsize: Tuple representing figure size (default is (12, 7)).
        """
        # Number of features to plot
        num_features = len(features)
        
        # Determine the number of rows and columns for subplots
        nrows = (num_features + 1) // 2  # Adjust the number of rows based on the features list
        ncols = 2 if num_features > 1 else 1  # Two columns if more than one feature
        
        # Create the figure and axes for subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1] * nrows))
        
        # Flatten axes array in case of a single subplot
        axes = axes.flatten() if num_features > 1 else [axes]
        
        # Loop through the features and create bar plots
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.countplot(data=self.df, x=feature, hue=hue, ax=ax)
            ax.set_title(f"Bar Plot for {feature}")
            ax.tick_params(axis='x', rotation=45)

            # Add count labels on top of each bar
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2, height + 0.5, int(height), 
                        ha="center", fontsize=10)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Show the plot
        plt.show()

    def scatter_plot(self, x_feature, y_feature, hue=None, figsize=(12, 7)):
        """
        Scatter plot for two continuous features
        """
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.df, x=x_feature, y=y_feature, hue=hue)
        plt.title(f"Scatter Plot: {x_feature} vs {y_feature}")
        plt.show()

    # def standardize_ip(self, ip):
    #     """
    #     Standardizes an IP address to a standard format.
    #     :param ip: The IP address to standardize.
    #     :return: Standardized IP address or None if invalid.
    #     """
    #     try:
    #         return str(ipaddress.ip_address(ip))
    #     except ValueError:
    #         return None  # Handle invalid IPs as needed

    # def standardize_ip_addresses(self):
    #     """
    #     Converts the 'ip_address' column in the DataFrame to a standard format.
    #     """
    #     # Apply the standardization
    #     self.df['ip_address'] = self.df['ip_address'].apply(self.standardize_ip)

    def merge_ip_addresses(self, Data2: pd.DataFrame):
        """
        Merges two DataFrames based on IP addresses and their ranges.
        :param Data2: DataFrame containing 'lower_bound_ip_address' and 'upper_bound_ip_address' columns.
        :return: Merged DataFrame with matched IP addresses.
        """
        # # Step 1: Standardize IP addresses
        # self.standardize_ip_addresses()  # Call without any arguments

        # # Standardize Data2 IP addresses
        # Data2['lower_bound_ip_address'] = Data2['lower_bound_ip_address'].apply(self.standardize_ip)
        # Data2['upper_bound_ip_address'] = Data2['upper_bound_ip_address'].apply(self.standardize_ip)

        # # Step 2: Drop rows with invalid IP addresses after standardization
        # self.df = self.df.dropna(subset=['ip_address'])
        # Data2 = Data2.dropna(subset=['lower_bound_ip_address', 'upper_bound_ip_address'])

        self.df['ip_address'] = self.df['ip_address'].astype(int)
        Data2['lower_bound_ip_address'] = Data2['lower_bound_ip_address'].astype(int)
        Data2['upper_bound_ip_address'] = Data2['upper_bound_ip_address'].astype(int)
         # Step 3: Convert IP addresses to integer for sorting and comparison
        self.df['ip_address'] = self.df['ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        Data2['lower_bound_ip_address'] = Data2['lower_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        Data2['upper_bound_ip_address'] = Data2['upper_bound_ip_address'].apply(lambda x: int(ipaddress.ip_address(x)))
        
        # Step 4: Sort both datasets based on the IP integer columns
        self.df = self.df.sort_values('ip_address')
        Data2 = Data2.sort_values('lower_bound_ip_address')

        # Step 5: Perform an asof merge to match IP addresses from self.df with the IP ranges in Data2
        merged_data = pd.merge_asof(self.df, Data2, 
                                     left_on='ip_address',   # Merge based on ip_address from self.df
                                     right_on='lower_bound_ip_address',  # Match with the lower bound of IP ranges in Data2
                                     direction='backward')  # 'backward' means match the closest lower bound

        # Step 6: Filter the merged dataset to include only rows where ip_address falls within the range (lower_bound, upper_bound)
        merged_data = merged_data[merged_data['ip_address'] <= merged_data['upper_bound_ip_address']]
        self.merged_data = merged_data
        return self.merged_data
    def geolocation_analysis_plots(self):
        # Check if merged_data is None before proceeding
        if self.merged_data is None:
            raise ValueError("merged_data is not initialized. Please run the merge_ip_addresses method first.")
        
        # Get the top 10 countries by user count
        top_countries = self.merged_data['country'].value_counts().nlargest(10)

        # Plot the distribution of users by the top 10 countries
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.merged_data[self.merged_data['country'].isin(top_countries.index)],
                      x='country', 
                      palette='coolwarm', 
                      order=top_countries.index)
        
        plt.title('Top 10 Countries by User Count', fontsize=15)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('User Count', fontsize=12)
        plt.xticks(rotation=45)  # Rotate x labels for better readability
        plt.show()
        purchase_value_by_country = self.merged_data.groupby('country')['purchase_value'].sum()
        top_purchase_countries = purchase_value_by_country.nlargest(10)  # Get the top 10 countries by purchase value

        sns.barplot(x=top_purchase_countries.index, 
                    y=top_purchase_countries.values, 
                    palette='viridis')
        
        plt.title('Top 10 Countries by Total Purchase Value', fontsize=15)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Total Purchase Value', fontsize=12)
        plt.xticks(rotation=45)  # Rotate x labels for better readability
        plt.show()
    def add_time_based_features(self):
        """Adds time-based features: 'hour_of_day' and 'day_of_week'."""
        # Extract day of the week from 'purchase_time' (Monday=0, Sunday=6)
        self.merged_data['purchase_day_of_week'] = self.merged_data['purchase_time'].dt.dayofweek

        self.merged_data['signup_hour'] = self.merged_data['signup_time'].dt.hour
        self.merged_data['signup_day'] = self.merged_data['signup_time'].dt.day
        self.merged_data['signup_month'] = self.merged_data['signup_time'].dt.month
        self.merged_data['purchase_day'] = self.merged_data['purchase_time'].dt.day
        self.merged_data['purchase_month'] = self.merged_data['purchase_time'].dt.month
        self.merged_data['purchase__hour'] = self.merged_data['purchase_time'].dt.hour
        
    
    def add_transaction_frequency_velocity(self):
        """Adds transaction frequency and velocity features."""
        # Sort the dataset by 'user_id' and 'purchase_time'
        self.merged_data = self.merged_data.sort_values(by=['user_id', 'purchase_time'])

        # Calculate time difference (velocity) between consecutive purchases for each user
        self.merged_data['time_diff'] = self.merged_data.groupby('user_id')['purchase_time'].diff().dt.total_seconds()

        # Calculate transaction count (frequency) for each user
        self.merged_data['transaction_count'] = self.merged_data.groupby('user_id').cumcount() + 1


        # Number of transactions per user
        self.merged_data['user_transaction_frequency'] = self.merged_data.groupby('user_id')['user_id'].transform('count')

        # Number of transactions per device
        self.merged_data['device_transaction_frequency'] = self.merged_data.groupby('device_id')['device_id'].transform('count')

        # Velocity of transactions per user and device
        self.merged_data['user_transaction_velocity'] = self.merged_data.groupby('user_id')['time_diff'].transform('mean')
        self.merged_data['device_transaction_velocity'] = self.merged_data.groupby('device_id')['time_diff'].transform('mean')

    def engineer_features(self):
        """Main function to engineer all necessary features."""
        self.add_time_based_features()
        self.add_transaction_frequency_velocity()
        print("Feature Engineering Complete!")
        return self.merged_data
    def encode_categorical_features(self):
        """Encodes categorical features using OneHotEncoder."""
        # Identify categorical features for encoding
        categorical_features = ['source', 'browser', 'sex']

        # Initialize the OneHotEncoder
        # Convert 'income_category' to dummy variables
        self.merged_data = pd.get_dummies(self.merged_data, columns=categorical_features, prefix='category', drop_first=False)
        # Display the resulting DataFrame
        self.merged_data
        return self.merged_data
    def Target_encoding(self):
       # Initialize the TargetEncoder with smoothing (adjust the value of smoothing as needed)
        target_encoder = ce.TargetEncoder(cols=['country'], smoothing=10)  # Here, 10 is the smoothing factor

        # Apply target encoding with smoothing
        self.merged_data['Country_encoded'] = target_encoder.fit_transform(self.merged_data['country'], self.merged_data['class'])
        target_encoder = ce.TargetEncoder(cols=['device_id'])
        self.merged_data['device_id_encoded'] = target_encoder.fit_transform(self.merged_data['device_id'], self.merged_data['class'])

        # Display the DataFrame with frequency encodings
        return self.merged_data
    def normalize_and_scale(self):
        """Normalizes and scales numerical features."""
        # Identify numerical features for scaling
        numerical_features = ['purchase_value','Country_encoded', 'age', 'purchase_day_of_week', 'signup_hour', 'signup_day', 'signup_month','purchase_day', 'purchase_month', 'purchase__hour']
        # Initialize the StandardScaler
        scaler = StandardScaler()
        # Fit and transform the numerical features
        self.merged_data[numerical_features] = scaler.fit_transform(self.merged_data[numerical_features])
        print("Numerical features scaled successfully.")
        return self.merged_data
    def scatter_plot1(self, x_feature, y_feature, hue=None, figsize=(12, 7)):
        """
        Scatter plot for two continuous features
        """
        plt.figure(figsize=figsize)
        sns.scatterplot(data=self.merged_data, x=x_feature, y=y_feature, hue=hue)
        plt.title(f"Scatter Plot: {x_feature} vs {y_feature}")
        plt.show()
    def country_distribution_per_class(self, class_col: str, country_col: str):
        """
        Plots the distribution of the top 10 countries per class.
        
        :param class_col: The column representing the class (e.g., 'fraud', 'non-fraud').
        :param country_col: The column representing the country.
        """
        # Ensure the specified columns exist in the DataFrame
        if class_col not in self.merged_data.columns or country_col not in self.merged_data.columns:
            raise ValueError(f"Columns '{class_col}' or '{country_col}' do not exist in the DataFrame.")

        # Count the occurrences of each country per class
        country_class_counts = self.merged_data.groupby([country_col, class_col]).size().reset_index(name='count')

        # Get the top 10 countries based on the total count
        top_countries = country_class_counts.groupby(country_col)['count'].sum().nlargest(10).index
        top_country_class_counts = country_class_counts[country_class_counts[country_col].isin(top_countries)]

        # Create a bar plot for the distribution
        plt.figure(figsize=(14, 7))
        sns.barplot(data=top_country_class_counts, x=country_col, y='count', hue=class_col, palette='viridis')
        
        plt.title('Top 10 Country Distribution per Class', fontsize=16)
        plt.xlabel('Country', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)  # Rotate x labels for better readability
        plt.legend(title=class_col)
        plt.show()

# Example usage
# Assuming you have an instance of FRAUD_EDA named fraud_eda and your DataFrame has 'country' and 'fraud_result' columns
# fraud_eda.country_distribution_per_class(class_col='fraud_result', country_col='country')

