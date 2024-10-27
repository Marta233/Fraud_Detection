import pandas as pd
import plotly.express as px
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
class Fraud_model:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        mlflow.sklearn.autolog()
    def fraud_preprocess(self):
        columns_to_convert = ['category_Ads', 'category_Direct', 'category_SEO', 'category_Chrome', 
                      'category_FireFox', 'category_IE', 'category_Opera', 'category_Safari', 
                      'category_F', 'category_M']

        self.df[columns_to_convert] = self.df[columns_to_convert].astype(int)
        self.df= self.df.drop(columns = ['user_id', 'country'], axis=1)
        return self.df
    def apply_smote(self):
        # Split data into training and test sets before applying SMOTE
        X = self.df.drop('class', axis=1)  # Drop the target column from the features
        y = self.df['class']  # Target variable
        
        # Split into training and testing sets (apply SMOTE only to training data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create a SMOTE instance to perform synthetic oversampling
        smote = SMOTE(random_state=42)
        # Apply SMOTE to the training data (not the test data)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        # Return the resampled training set and the untouched test set
        return X_train_res, y_train_res, X_test, y_test
    def mode_logistic_rehgression(self):
        # Ensure the necessary columns exist
       # Convert specific columns from boolean to integers
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()
        X = self.df.drop('class', axis=1)  # Assuming 'fraud_bool' is the target variable
        y = self.df['class']
        # Training the logistic regression model
        model = LogisticRegression()
        model.fit(X_train_res, y_train_res)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return model
    def model_decision_tree(self):
        # Ensure the necessary columns exist
       # Convert specific columns from boolean to integers
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()
        X = self.df.drop('class', axis=1)  # Assuming 'fraud_bool' is the target variable
        y = self.df['class']
        # Training the logistic regression model
        model = DecisionTreeClassifier()
        model.fit(X_train_res, y_train_res)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return model
    def model_random_forest(self):
        # Ensure the necessary columns exist
       # Convert specific columns from boolean to integers
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()
        X = self.df.drop('class', axis=1)  # Assuming 'fraud_bool' is the target variable
        y = self.df['class']
        # Training the logistic regression model
        model = RandomForestClassifier()
        model.fit(X_train_res, y_train_res)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return model
    def model_gradient_boosting(self):
        # Ensure the necessary columns exist
       # Convert specific columns from boolean to integers
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()
        X = self.df.drop('class', axis=1)  # Assuming 'fraud_bool' is the target variable
        y = self.df['class']
        # Training the logistic regression model
        model = GradientBoostingClassifier()
        model.fit(X_train_res, y_train_res)

        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluating the model
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy: {accuracy:.2f}")
        return model
    def model_mlp(self):
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()
        
        # Build MLP model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=X_train_res.shape[1]))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test)[1]
        print(f"MLP Accuracy: {accuracy:.2f}")
        return model

    def model_cnn(self):
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()

        # Reshape for CNN (adding a dimension)
        X_train_res = np.expand_dims(X_train_res, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        # Build CNN model
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_res.shape[1], 1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test)[1]
        print(f"CNN Accuracy: {accuracy:.2f}")
        return model

    def model_rnn(self):
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()

        # Reshape for RNN (adding a dimension)
        X_train_res = np.expand_dims(X_train_res, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        # Build RNN model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train_res.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test)[1]
        print(f"RNN Accuracy: {accuracy:.2f}")
        return model

    def model_lstm(self):
        X_train_res, y_train_res, X_test, y_test = self.apply_smote()

        # Reshape for LSTM (adding a dimension)
        X_train_res = np.expand_dims(X_train_res, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(X_train_res.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(X_train_res, y_train_res, epochs=10, batch_size=32, verbose=1)

        # Evaluate the model
        accuracy = model.evaluate(X_test, y_test)[1]
        print(f"LSTM Accuracy: {accuracy:.2f}")
        return model
    



















































    import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

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
        accuracy, precision, recall, f1, auc = self.evaluate_model(y_test, y_pred)

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

    def predict_with_threshold(self, model, X_test):
        """Standardized prediction method for binary classification."""
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = model.predict(X_test)
        return (y_pred_prob > 0.5).astype("int32")

    def model_logistic_regression(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Logistic Regression", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = LogisticRegression()
            model.fit(X_train_res, y_train_res)
            y_pred = self.predict_with_threshold(model, X_test)
            hyperparameters = {"C": model.C, "solver": model.solver}
            duration = time.time() - start_time
            self.log_metrics("Logistic Regression", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model

    def model_decision_tree(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Decision Tree", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = DecisionTreeClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = self.predict_with_threshold(model, X_test)
            hyperparameters = {"max_depth": model.max_depth, "min_samples_split": model.min_samples_split}
            duration = time.time() - start_time
            self.log_metrics("Decision Tree", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model

    def model_random_forest(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Random Forest", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train_res, y_train_res)
            y_pred = self.predict_with_threshold(model, X_test)
            hyperparameters = {"n_estimators": 100, "max_depth": model.max_depth}
            duration = time.time() - start_time
            self.log_metrics("Random Forest", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model

    def model_gradient_boosting(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - Gradient Boosting", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()
            model = GradientBoostingClassifier()
            model.fit(X_train_res, y_train_res)
            y_pred = self.predict_with_threshold(model, X_test)
            hyperparameters = {"n_estimators": model.n_estimators, "learning_rate": model.learning_rate}
            duration = time.time() - start_time
            self.log_metrics("Gradient Boosting", X_train_res, y_train_res, y_test, y_pred, hyperparameters, duration)
            return model

    def log_deep_learning_params(self, model_name, model, epochs, batch_size, X_train_res, y_train_res, y_test, y_pred, duration):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("layers", len(model.layers))
        
        for i, layer in enumerate(model.layers):
            mlflow.log_param(f"layer_{i}_type", layer.__class__.__name__)
            mlflow.log_param(f"layer_{i}_units", getattr(layer, "units", "N/A"))
            mlflow.log_param(f"layer_{i}_activation", getattr(layer, "activation", "N/A"))

        optimizer = model.optimizer
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("learning_rate", optimizer.learning_rate.numpy())

        self.log_metrics(model_name, X_train_res, y_train_res, y_test, y_pred, None, duration)

        mlflow.keras.log_model(model, f"{model_name}_model")

    def model_mlp(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - MLP", nested=True):
            X_train_res, y_train_res, X_test, y_test = self.apply_smote()

            model = Sequential([
                Dense(64, activation='relu', input_dim=X_train_res.shape[1]),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

            epochs = 10
            batch_size = 32
            model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=batch_size, verbose=1)

            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            duration = time.time() - start_time
            self.log_deep_learning_params("MLP", model, epochs, batch_size, X_train_res, y_train_res, y_test, y_pred, duration)

            mlflow.keras.log_model(model, "MLP_model")
        return model

    def fraud_preprocess_lstm_rnn(self, sequence_length=10):
        X = self.df.drop('class', axis=1).values
        y = self.df['class'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        generator = TimeseriesGenerator(X_train, y_train, length=sequence_length, batch_size=32)
        test_generator = TimeseriesGenerator(X_test, y_test, length=sequence_length, batch_size=32)

        return generator, test_generator, y_test

    def model_lstm(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - LSTM", nested=True):
            generator, test_generator, y_test = self.fraud_preprocess_lstm_rnn()

            model = Sequential([
                LSTM(64, input_shape=(generator.length, generator.data.shape[1]), return_sequences=True),
                LSTM(32),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

            epochs = 10
            batch_size = 32
            model.fit(generator, epochs=epochs, verbose=1)

            y_pred = (model.predict(test_generator) > 0.5).astype("int32")
            duration = time.time() - start_time
            self.log_deep_learning_params("LSTM", model, epochs, batch_size, generator.data, generator.targets, y_test, y_pred, duration)

            mlflow.keras.log_model(model, "LSTM_model")
        return model

    def model_rnn(self):
        start_time = time.time()
        with mlflow.start_run(run_name=f"{self.dataset_name} - RNN", nested=True):
            generator, test_generator, y_test = self.fraud_preprocess_lstm_rnn()

            model = Sequential([
                SimpleRNN(64, input_shape=(generator.length, generator.data.shape[1]), return_sequences=True),
                SimpleRNN(32),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

            epochs = 10
            batch_size = 32
            model.fit(generator, epochs=epochs, verbose=1)

            y_pred = (model.predict(test_generator) > 0.5).astype("int32")
            duration = time.time() - start_time
            self.log_deep_learning_params("RNN", model, epochs, batch_size, generator.data, generator.targets, y_test, y_pred, duration)

            mlflow.keras.log_model(model, "RNN_model")
        return model

    def display_metrics_table(self):
        return self.metrics_df