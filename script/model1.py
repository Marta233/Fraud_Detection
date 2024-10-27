import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time  # Importing time module for duration calculation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam
import time
from sklearn.metrics import accuracy_score

class FraudModel:
    def __init__(self, df: pd.DataFrame, dataset_name: str, source: str):
        self.df = df
        self.dataset_name = dataset_name
        self.source = source
        self.metrics_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"])
        mlflow.sklearn.autolog()

    def fraud_preprocess(self):
        columns_to_convert = ['category_Ads', 'category_Direct', 'category_SEO', 'category_Chrome', 
                              'category_FireFox', 'category_IE', 'category_Opera', 'category_Safari', 
                              'category_F', 'category_M']
        self.df[columns_to_convert] = self.df[columns_to_convert].astype(int)
        self.df = self.df.drop(columns=['user_id', 'country'], axis=1)
        return self.df

    def apply_smote(self):
        X = self.df.drop('class', axis=1)
        y = self.df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        return X_train_res, y_train_res, X_test, y_test

    def evaluate_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        return accuracy, precision, recall, f1, auc

    def log_metrics(self, model_name, X_train_res, y_train_res, y_test, y_pred, hyperparameters=None, duration=None):
        # Evaluate model
        accuracy, precision, recall, f1, auc = self.evaluate_model(y_test, y_pred)

        # Append the metrics to the metrics_df
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

        mlflow.log_param("dataset_name", self.dataset_name)
        mlflow.log_param("source", self.source)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("duration", duration)
        
        # # Adding input example for model logging
        # input_example = X_train_res[:1]  # Use first row of training data as example input

        # Log the model with input example
        mlflow.sklearn.log_model(model_name, f"{model_name.lower()}_model")

        # Log hyperparameters if provided
        if hyperparameters:
            for key, value in hyperparameters.items():
                mlflow.log_param(key, value)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_param("train_size", len(y_train_res))
        mlflow.log_param("feature_count", X_train_res.shape[1])
    def model_logistic_regression(self):
        start_time = time.time()  # Start time for duration calculation
        with mlflow.start_run(run_name=f"{self.dataset_name} - Logistic Regression", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = LogisticRegression()
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            hyperparameters = {"C": model.C, "solver": model.solver}
            duration = time.time() - start_time  # Calculate duration
            # Evaluating the model
            accuracy = model.score(X_test, y_test)
            print(f"Accuracy: {accuracy:.2f}")
            self.log_metrics("Logistic Regression",  X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model
    def model_decision_tree(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Decision Tree", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = DecisionTreeClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            hyperparameters = {"max_depth": model.max_depth, "min_samples_split": model.min_samples_split}
            duration = time.time() - start_time
            # Evaluating the model
            accuracy = model.score(X_test, y_test)
            print(f"Accuracy: {accuracy:.2f}")
            self.log_metrics("Decision Tree", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model
    def model_random_forest(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Random Forest", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            hyperparameters = {"n_estimators": 50, "max_depth": model.max_depth}
            duration = time.time() - start_time
            # Evaluating the model
            accuracy = model.score(X_test, y_test)
            print(f"Accuracy: {accuracy:.2f}")
            self.log_metrics("Random Forest",  X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model

    def model_gradient_boosting(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Gradient Boosting", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = GradientBoostingClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test)
            hyperparameters = {"n_estimators": model.n_estimators, "learning_rate": model.learning_rate}
            duration = time.time() - start_time
            # Evaluating the model
            accuracy = model.score(X_test, y_test)
            print(f"Accuracy: {accuracy:.2f}")
            self.log_metrics("Gradient Boosting", X_train_res, y_train_res,  y_test, y_pred, hyperparameters, duration)
            return model
    # Function for logging deep learning model parameters and metrics
    def log_deep_learning_params(self, model_name, model, epochs, batch_size, X_train_res, y_train_res, y_test, y_pred, duration):
        # Log model structure and hyperparameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("layers", len(model.layers))
        
        for i, layer in enumerate(model.layers):
            mlflow.log_param(f"layer_{i}_type", layer.__class__.__name__)
            mlflow.log_param(f"layer_{i}_units", getattr(layer, "units", "N/A"))
            mlflow.log_param(f"layer_{i}_activation", getattr(layer, "activation", "N/A"))

        # Log optimizer settings
        optimizer = model.optimizer
        mlflow.log_param("optimizer", optimizer.__class__.__name__)  # Change to this line
        mlflow.log_param("learning_rate", optimizer.learning_rate.numpy())

        # Log metrics
        self.log_metrics(model_name, X_train_res, y_train_res, y_test, y_pred, None, duration)

        # # Log the Keras model with an input example
        # input_example = np.array(X_train_res[:1])  # Use the first example of the training set
        mlflow.keras.log_model(model, f"{model_name}_model")



    def model_mlp(self):
        start_time = time.time()
        try:
            with mlflow.start_run(run_name=f"{self.dataset_name} - MLP", nested=True):
                X_train_res, y_train_res, X_test, y_test = self.apply_smote()

                # MLP model setup
                model = Sequential([
                    Dense(64, activation='relu', input_dim=X_train_res.shape[1]),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')  # Binary classification
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

                # Model training
                epochs = 10
                batch_size = 32
                model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=batch_size, verbose=1)

                # Prediction
                y_pred = (model.predict(X_test) > 0.5).astype("int32")
                duration = time.time() - start_time
                # Evaluating the model
                accuracy = model.score(X_test, y_test)
                print(f"Accuracy: {accuracy:.2f}")
                # Log deep learning parameters and metrics
                self.log_deep_learning_params("MLP", model, epochs, batch_size, X_train_res, y_train_res, y_test, y_pred, duration)

                # Log the Keras model to MLflow
                mlflow.keras.log_model(model, "MLP_model")

        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.end_run(status="FAILED")  # End the run as failed if an exception occurs
        else:
            mlflow.end_run(status="FINISHED")  # End the run as finished if successful

        return model
    def apply_smote11(self, X, y):
        # Reshape X for SMOTE (flattening the time-series data)
        n_samples, n_time_steps, n_features = X.shape
        X_reshaped = X.reshape(n_samples, n_time_steps * n_features)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)

        # Reshape back to the original time-series format
        n_resampled_samples = X_resampled.shape[0]
        X_resampled = X_resampled.reshape(n_resampled_samples, n_time_steps, n_features)

        return X_resampled, y_resampled

    def fraud_preprocess_lstm_rnn(self, sequence_length=10):
        X = self.df.drop('class', axis=1).values
        y = self.df['class'].values

        assert len(X) == len(y), "Mismatch in number of samples between X and y"

        # Split the data first
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Calculate the number of samples we can create with the given sequence length
        n_samples = len(X_train) // sequence_length
        if n_samples < 1:
            raise ValueError("Sequence length must be less than the number of training samples.")

        # Reshape X_train for LSTM
        X_train = X_train[:n_samples * sequence_length].reshape(-1, sequence_length, X_train.shape[1])  # (n_samples, sequence_length, n_features)
        
        # Ensure y_train has the correct shape
        y_train = y_train[:n_samples * sequence_length].reshape(-1, sequence_length)[:, -1]  # Last value as label

        # Apply SMOTE
        X_train_res, y_train_res = self.apply_smote11(X_train, y_train)

        # Create generators
        train_generator = TimeseriesGenerator(X_train_res, y_train_res, length=sequence_length, batch_size=32)
        test_generator = TimeseriesGenerator(X_test, y_test, length=sequence_length, batch_size=32)

        return train_generator, test_generator, y_test
    def model_lstm(self):
        start_time = time.time()
        model = None  # Initialize model variable to handle UnboundLocalError
        try:
            with mlflow.start_run(run_name=f"{self.dataset_name} - LSTM", nested=True):
                # Apply SMOTE to handle class imbalance
                X_train_res, y_train_res, X_test, y_test = self.apply_smote()
                
                # Convert DataFrames to NumPy arrays if necessary and reshape input for LSTM
                X_train_res = np.array(X_train_res)
                X_test = np.array(X_test)
                
                # Reshape input for LSTM: (samples, time steps, features)
                X_train_res = X_train_res.reshape((X_train_res.shape[0], 1, X_train_res.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                # LSTM model setup
                model = Sequential([
                    LSTM(64, activation='relu', input_shape=(X_train_res.shape[1], X_train_res.shape[2])),  # Input shape
                    Dense(32, activation='relu'),  # Hidden layer
                    Dense(1, activation='sigmoid')  # Output layer for binary classification
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), 
                            loss='binary_crossentropy', 
                            metrics=['accuracy'])

                # Model training
                epochs = 10
                batch_size = 32
                model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=batch_size, verbose=1)

                # Prediction
                y_pred = (model.predict(X_test) > 0.5).astype("int32")

                # Evaluating the model
                accuracy = accuracy_score(y_test, y_pred)
                duration = time.time() - start_time
                
                print(f"Accuracy: {accuracy:.2f}")
                
                # Log deep learning parameters and metrics
                self.log_deep_learning_params(
                    "LSTM", model, epochs, batch_size, X_train_res, y_train_res, y_test, y_pred, duration
                )

                # Log the Keras model to MLflow
                mlflow.keras.log_model(model, "LSTM_model")

        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.end_run(status="FAILED")  # End the run as failed if an exception occurs
        else:
            mlflow.end_run(status="FINISHED")  # End the run as finished if successful

        return model
    def model_rnn(self):
        start_time = time.time()
        model = None  # Initialize model variable to handle UnboundLocalError
        try:
            with mlflow.start_run(run_name=f"{self.dataset_name} - RNN", nested=True):
                # Apply SMOTE to handle class imbalance
                X_train_res, y_train_res, X_test, y_test = self.apply_smote()
                
                # Convert DataFrames to NumPy arrays if necessary and reshape input for RNN
                X_train_res = np.array(X_train_res)
                X_test = np.array(X_test)
                
                # Reshape input for RNN: (samples, time steps, features)
                X_train_res = X_train_res.reshape((X_train_res.shape[0], 1, X_train_res.shape[1]))
                X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                # RNN model setup
                model = Sequential([
                    SimpleRNN(64, activation='relu', input_shape=(X_train_res.shape[1], X_train_res.shape[2])),  # Input shape
                    Dense(32, activation='relu'),  # Hidden layer
                    Dense(1, activation='sigmoid')  # Output layer for binary classification
                ])
                model.compile(optimizer=Adam(learning_rate=0.001), 
                            loss='binary_crossentropy', 
                            metrics=['accuracy'])

                # Model training
                epochs = 10
                batch_size = 32
                model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=batch_size, verbose=1)

                # Prediction
                y_pred = (model.predict(X_test) > 0.5).astype("int32")

                # Evaluating the model
                accuracy = accuracy_score(y_test, y_pred)
                duration = time.time() - start_time
                
                print(f"Accuracy: {accuracy:.2f}")
                
                # Log deep learning parameters and metrics
                self.log_deep_learning_params(
                    "RNN", model, epochs, batch_size, X_train_res, y_train_res, y_test, y_pred, duration
                )

                # Log the Keras model to MLflow
                mlflow.keras.log_model(model, "RNN_model")

        except Exception as e:
            print(f"An error occurred: {e}")
            mlflow.end_run(status="FAILED")  # End the run as failed if an exception occurs
        else:
            mlflow.end_run(status="FINISHED")  # End the run as finished if successful

        return model
    def display_metrics_table(self):
        return self.metrics_df





