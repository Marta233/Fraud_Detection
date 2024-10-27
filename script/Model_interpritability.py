import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib  # For loading scikit-learn models
import matplotlib.pyplot as plt
import logging
import mlflow  # Import MLflow for logging control

# Set the logging level for MLflow to suppress info messages
mlflow_logger = logging.getLogger("mlflow")
mlflow_logger.setLevel(logging.WARNING)

class ModelTrainer:
    def __init__(self, df):
        """
        Initialize the ModelTrainer with a DataFrame.

        Args:
        - df: pandas DataFrame containing features and target variable.
        """
        self.df = df
        self.model = None

    def load_model(self, model_name, model_path):
        self.model = joblib.load(model_path)
        print(f"Model {model_name} loaded successfully.")

    def explain_with_lime(self, X_train, X_test):
        """Explain predictions using LIME."""
        # Create LIME Explainer
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            mode='classification',
            feature_names=X_train.columns,
            class_names=['0', '1']
        )
        # Check if X_test is a DataFrame
        if isinstance(X_test, pd.DataFrame):
            # Get the first instance as a 1D array
            instance = X_test.iloc[0].to_numpy()  # This should now work correctly
        else:
            raise ValueError("X_test should be a pandas DataFrame.")

        # Generate explanation
        exp = explainer.explain_instance(instance, self.model.predict_proba, num_features=30)

        # Plot local explanation
        plt = exp.as_pyplot_figure()
        plt.tight_layout()
        exp.show_in_notebook(show_table=True)
    def explain_with_shap(self, X_train_res, sample_size=100):
        """Create a SHAP explainer using a sample of the training data."""
        if sample_size is not None:
            # Sample the training data if sample_size is provided
            X_train_res = X_train_res.sample(n=min(sample_size, len(X_train_res)), random_state=42)

        self.explainer = shap.KernelExplainer(self.model.predict_proba, X_train_res)
        print("SHAP explainer created.")
    def calculate_shap_values(self, X_test, sample_size=100):
        """Calculate SHAP values for the test set with optional sampling."""
        if not hasattr(self, 'explainer'):
            raise ValueError("SHAP explainer not created. Call explain_with_shap first.")
        
        # Sample the test set if sample_size is provided
        if sample_size is not None:
            sample_size = min(sample_size, len(X_test))  # Ensure we don't sample more than available
            sampled_indices = np.random.choice(X_test.index, size=sample_size, replace=False)
            X_test_sampled = X_test.loc[sampled_indices]

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_test_sampled)
        
        print("SHAP values calculated.")
        return shap_values, sampled_indices

    def plot_shap_summary(self, shap_values, X_test):
        """Plot summary of SHAP values."""
        shap.summary_plot(shap_values, X_test)
    def plot_force(self, shap_values, X_test, index=0):
        """Plot SHAP Force Plot for a single prediction."""
        if not hasattr(self, 'explainer'):
            raise ValueError("SHAP explainer not created. Call explain_with_shap first.")
        
        # Generate the force plot for a single instance
        shap.initjs()  # Initialize JS for Jupyter Notebook
        shap.force_plot(self.explainer.expected_value, shap_values[index], X_test.iloc[index])
    
    def plot_dependence(self, shap_values, X_test, feature_name):
        """Plot SHAP Dependence Plot for a specific feature."""
        shap.dependence_plot(feature_name, shap_values, X_test)