import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import shap
from lime.lime_tabular import LimeTabularExplainer
import joblib
import matplotlib.pyplot as plt
import logging

# Set the logging level for MLflow to suppress info messages
mlflow_logger = logging.getLogger("mlflow")
mlflow_logger.setLevel(logging.WARNING)

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.X_train_res = None
        self.X_test = None
        self.y_train_res = None
        self.y_test = None

    def train_model(self):
        """Train Random Forest model with SMOTE."""
        X = self.df.drop(columns=['class'])
        y = self.df['class']

        # Split data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        self.X_train_res, self.y_train_res = smote.fit_resample(self.X_train, self.y_train)

        # Train the model
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train_res, self.y_train_res)

        # Optionally, save the model
        joblib.dump(self.model, 'random_forest_model.joblib')
        print("Random Forest model trained and saved.")

        return self.X_train_res, self.X_test, self.y_train_res, self.y_test

    def explain_with_lime(self):
        """Explain predictions using LIME."""
        # Create LIME Explainer
        explainer = LimeTabularExplainer(
            training_data=np.array(self.X_train_res),
            mode='classification',
            feature_names=self.X_train_res.columns,
            class_names=['0', '1']
        )

        # Check if X_test is a DataFrame
        if isinstance(self.X_test, pd.DataFrame):
            # Get the first instance as a 1D array
            instance = self.X_test.iloc[0].to_numpy()  # This should now work correctly
        else:
            raise ValueError("X_test should be a pandas DataFrame.")

        # Generate explanation
        exp = explainer.explain_instance(instance, self.model.predict_proba, num_features=30)

        # Plot local explanation
        plt = exp.as_pyplot_figure()
        plt.tight_layout()
        exp.show_in_notebook(show_table=True)

    def explain_with_shap(self):
        """Explain predictions using SHAP."""
        # Sample 100 rows from X_test
        sample_size = min(100, len(self.X_test))  # Ensure not to exceed available data
        X_test_sample = self.X_test.sample(n=sample_size, random_state=42)

        explainer = shap.Explainer(self.model, self.X_train_res)
        shap_values = explainer(X_test_sample)
        return shap_values, explainer  # Return both shap values and explainer

    def shap_summary_plot(self, shap_values):
        """Create a SHAP summary plot."""
        shap.summary_plot(shap_values, self.X_test)

    def shap_force_plot(self, shap_values, explainer, index=0):
        """Create a SHAP force plot."""
        # Ensure shap.initjs() for notebook environments
        shap.initjs()
        plt.figure()
        shap.force_plot(explainer.expected_value, shap_values[index], self.X_test.iloc[index, :], matplotlib=True)
        plt.show()

    def shap_dependence_plot(self, shap_values, feature_name):
        """Create a SHAP dependence plot."""
        shap.dependence_plot(feature_name, shap_values, self.X_test)
