import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.combine import SMOTETomek
import os
import time
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn=None, **kwargs):  # Corrected here
        self.build_fn = build_fn
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        self.model = self.build_fn(**self.kwargs)
        self.model.fit(X, y, epochs=self.kwargs.get('epochs', 10), batch_size=self.kwargs.get('batch_size', 32))
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def predict_proba(self, X):
        return self.model.predict(X)
class FraudModel:
    def __init__(self, df, dataset_name, source):
        self.df = df
        self.dataset_name = dataset_name
        self.source = source
        self.metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"])
        mlflow.sklearn.autolog()
        mlflow.tensorflow.autolog()
    def fraud_preprocess(self):
        columns_to_convert = ['category_Ads', 'category_Direct', 'category_SEO', 'category_Chrome', 
                            'category_FireFox', 'category_IE', 'category_Opera', 'category_Safari', 
                            'category_F', 'category_M']
        self.df[columns_to_convert] = self.df[columns_to_convert].astype(int)
        self.df = self.df.drop(columns=['user_id', 'country'], axis=1)
        return self.df

    def apply_smote(self, minority_percentage=0.5):
        X = self.df.drop('class', axis=1)
        y = self.df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if X_train.isnull().any().any() or y_train.isnull().any():
            raise ValueError("Input data contains missing values. Please handle them before applying SMOTE.")

        majority_class_count = np.sum(y_train == 0)
        desired_minority_count = int(minority_percentage * majority_class_count)
        smote_tomek = SMOTETomek(sampling_strategy={1: desired_minority_count}, random_state=42)
        X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

        print("New class distribution in training set:", np.bincount(y_train_res))
        return X_train_res, y_train_res, X_test, y_test

    def evaluate_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        return accuracy, precision, recall, f1, auc, class_report

    def log_metrics(self, model_name, X_train_res, y_train_res, y_test, y_pred, hyperparameters=None, duration=None):
        accuracy, precision, recall, f1, auc, class_report = self.evaluate_model(y_test, y_pred)
        new_row = pd.DataFrame({
            "Model": [model_name],
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1],
            "AUC": [auc]
        })
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)

        print(f"Logging metrics for model: {model_name}")
        print(f"Dataset name: {self.dataset_name}, Source: {self.source}")
        print("\nClassification Report:\n", class_report)

        mlflow.log_param("dataset_name", self.dataset_name)
        mlflow.log_param("source", self.source)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("duration", duration)

        if hyperparameters:
            for key, value in hyperparameters.items():
                mlflow.log_param(key, value)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_param("train_size", len(y_train_res))
        mlflow.log_param("feature_count", X_train_res.shape[1])

    def save_model(self, model, model_name):
        print(f"Attempting to save model of type: {type(model)}")
        """Save model locally and log it with MLflow."""
        model_dir = 'models'  # Adjust if necessary
        os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist
        model_path = os.path.join(model_dir, f"{model_name}.pkl")

        try:
            # For sklearn models
            if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, MLPClassifier)):
                joblib.dump(model, model_path)  # Save locally
                print(f"Model {model_name} saved at {model_path}")
                mlflow.sklearn.log_model(model, model_name)  # Log to MLflow
            # For Keras models
            elif isinstance(model, Sequential):
                model.save(os.path.join(model_dir, f"{model_name}.h5"))  # Save locally
                print(f"Keras model {model_name} saved at {model_path}.h5")
                mlflow.tensorflow.log_model(model, model_name)  # Log to MLflow
            else:
                print("Model type not recognized for saving.")
        except Exception as e:
            print(f"Error saving model: {e}")
    def display_metrics(self):
        print("\nEvaluation Metrics:")
        print(self.metrics_df)

    def model_decision_tree(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Decision Tree", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            # Initialize the model without hyperparameter tuning
            pipeline = Pipeline([
                ('decision_tree', DecisionTreeClassifier())  # Default parameters
            ])
            # Fit the model directly
            pipeline.fit(X_train_res, y_train_res)
            # Make predictions
            y_pred = pipeline.predict(X_test)
            # Log metrics and duration
            duration = time.time() - start_time
            self.log_metrics("Decision Tree", X_train_res, y_train_res, y_test, y_pred, {}, duration)
            # Attempt to save the model
            print("Attempting to save the model...")
            self.save_model(pipeline.named_steps['decision_tree'], f"{self.dataset_name}_Decision_Tree")
            print("Model save attempted.")
            print("\nClassification Report for Decision Tree:\n", classification_report(y_test, y_pred))
            return pipeline

    def model_random_forest(self):
        start_time = time.time()
        try:
            with mlflow.start_run(run_name=f"{self.dataset_name} - Random Forest", nested=True):
                X_train_res, y_train_res, X_test, y_test = self.apply_smote()
                
                # Initialize the model with default parameters
                pipeline = Pipeline([
                    ('random_forest', RandomForestClassifier(random_state=42, n_estimators=50))  # Default parameters
                ])
                
                # Fit the model directly
                pipeline.fit(X_train_res, y_train_res)

                # Make predictions
                y_pred = pipeline.predict(X_test)

                # Log metrics and duration
                hyperparameters = {}  # No hyperparameters to log
                duration = time.time() - start_time
                self.log_metrics("Random Forest", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
                # Save the actual model from the pipeline
                random_forest_model = pipeline.named_steps['random_forest']
                print("Attempting to save random forest model...")
                self.save_model(random_forest_model, f"{self.dataset_name}_Random_Forest")
                print("Model save attempted.")
                print("\nClassification Report for Random Forest:\n", classification_report(y_test, y_pred))
        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")

    def model_mlp(self):
        start_time = time.time()
        try:
            with mlflow.start_run(run_name=f"{self.dataset_name} - MLP", nested=True):
                X_train_res, y_train_res, X_test, y_test = self.apply_smote()
                
                # Initialize the MLP model with default parameters
                mlp_pipeline = Pipeline([
                    ('mlp', MLPClassifier(max_iter=1000, random_state=42))  # Default parameters
                ])
                
                # Fit the model directly
                mlp_pipeline.fit(X_train_res, y_train_res)

                # Make predictions
                y_pred_probs = mlp_pipeline.predict_proba(X_test)[:, 1]  # Use predict_proba
                y_pred = (y_pred_probs > 0.5).astype(int)

                # Log metrics and duration
                hyperparameters = {}  # No hyperparameters to log
                duration = time.time() - start_time
                self.log_metrics("MLP", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)

                # Save the actual MLP model from the pipeline
                mlp_model = mlp_pipeline.named_steps['mlp']
                print("Attempting to save MLP model...")
                self.save_model(mlp_model, f"{self.dataset_name}_MLP")
                print("Model save attempted.")

                print("\nClassification Report for MLP:\n", classification_report(y_test, y_pred))
        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")
        
        return mlp_pipeline

    def model_lstm(self):
        start_time = time.time()
        lstm_pipeline = None  # Initialize lstm_pipeline
        try:
            with mlflow.start_run(run_name=f"{self.dataset_name} - LSTM", nested=True):
                X_train_res, y_train_res, X_test, y_test = self.apply_smote()
                X_train_res = X_train_res.values.reshape((X_train_res.shape[0], X_train_res.shape[1], 1))
                X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

                # Create the LSTM model building function
                def create_lstm_model():
                    model = Sequential()
                    model.add(LSTM(50, activation='relu', input_shape=(X_train_res.shape[1], 1)))
                    model.add(Dropout(0.2))
                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    return model
                lstm_pipeline = Pipeline([
                    ('lstm_classifier', KerasClassifierWrapper(build_fn=create_lstm_model, epochs=10, batch_size=32))  # Pass arguments here
                ])

                # Fit the model directly
                lstm_pipeline.fit(X_train_res, y_train_res)

                # Make predictions
                y_pred = lstm_pipeline.predict(X_test)

                # Log metrics and duration
                hyperparameters = {'hidden_units': 50, 'batch_size': 32, 'epochs': 50}
                duration = time.time() - start_time
                self.log_metrics("LSTM", X_train_res, y_train_res, y_test, y_pred.flatten(), hyperparameters, duration)

                # Save the entire pipeline
                print("Attempting to save LSTM model pipeline...")
                self.save_model(lstm_pipeline, f"{self.dataset_name}_LSTM_Pipeline")
                print("Model save attempted.")
        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")
        
        return lstm_pipeline  # This will return None if an error occurred

    def display_metrics_table(self):
        return self.metrics_df