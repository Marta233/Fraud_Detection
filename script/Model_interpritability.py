import sys
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek  # Correct import for SMOTETomek
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FraudDetectionModel:
    def __init__(self, data_path):
        # Load data
        logging.info("Loading data from %s", data_path)
        self.data = pd.read_csv(data_path, index_col=0)
        self.data = self.data.drop(columns=['country'])
        
        # Prepare features and target
        self.X = self.data.drop(columns=['class'])
        self.y = self.data['class']
        
        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        self.model = RandomForestClassifier(random_state=42)
        logging.info("Initialized RandomForestClassifier model.")

    def apply_smote(self):
        """Apply SMOTETomek to balance the dataset."""
        logging.info("Applying SMOTETomek to balance the dataset...")
        smote_tomek = SMOTETomek(random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote_tomek.fit_resample(self.X_train, self.y_train)
        logging.info(f"Original training set size: {self.X_train.shape[0]} | Resampled training set size: {self.X_train_resampled.shape[0]}")

    def train_model(self):
        """Train the Random Forest model on training data."""
        logging.info("Training the model...")
        self.model.fit(self.X_train_resampled, self.y_train_resampled)
        logging.info("Model trained successfully.")

    def evaluate_model(self):
        """Evaluate the model and print accuracy and classification report."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        logging.info("Accuracy: %f", accuracy)
        logging.info("\nClassification Report:\n%s", classification_report(self.y_test, y_pred))

    def explain_with_shap(self, sample_size=100):
        """Explain predictions using SHAP."""
        logging.info("Explaining predictions using SHAP...")
        # Sample the test set
        X_test_sample = self.X_test.sample(sample_size, random_state=42)

        # Create SHAP explainer
        explainer = shap.Explainer(self.model, self.X_train_resampled)  # Use resampled training data
        shap_values = explainer(X_test_sample, check_additivity=False)

        # Generate summary plot
        self.shap_summary_plot(shap_values, X_test_sample)

        # Generate force plot for the first instance
        self.shap_force_plot(shap_values, explainer, index=0, X_test_sample=X_test_sample)

    def shap_summary_plot(self, shap_values, X_test_sample):
        """Create a SHAP summary plot."""
        logging.info("Creating SHAP Summary Plot...")
        shap.summary_plot(shap_values[..., 1], X_test_sample)  # For fraud class

    def shap_force_plot(self, shap_values, explainer, index=0, X_test_sample=None):
        """Create a SHAP force plot for a specific instance."""
        if X_test_sample is not None:
            shap.initjs()
            plt.figure()
            shap.force_plot(
                explainer.expected_value[1],
                shap_values.values[index, :, 1],
                X_test_sample.iloc[index],
                matplotlib=True
            )
            plt.show()
        else:
            logging.error("X_test_sample must be provided for the force plot.")

    def explain_with_lime(self):
        """Explain predictions using LIME."""
        logging.info("Explaining predictions using LIME...")
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            training_data=np.array(self.X_train_resampled),
            mode='classification',
            feature_names=self.X_train.columns,
            class_names=['0', '1']
        )

        # Get the first instance from the test set
        instance = self.X_test.iloc[0].to_numpy()
        
        # Generate explanation
        exp = explainer.explain_instance(instance, self.model.predict_proba, num_features=30)
        
        # Plot local explanation
        plt = exp.as_pyplot_figure()
        plt.tight_layout()
        exp.show_in_notebook(show_table=True)
