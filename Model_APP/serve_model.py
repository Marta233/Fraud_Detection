from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # This will allow all domains to access your API
class FraudModel:
    def __init__(self):
        self.model = self.load_model()  # Load your trained model here
        self.scaler = StandardScaler()

    def load_model(self):
        # Load the model from a file
        model = joblib.load('./Fraud_Random_Forest.pkl')
        self.feature_order = model.feature_names_in_  # Extract feature order directly from the model
        return model

    def add_time_based_features(self, input_data):
        """Adds time-based features to input data."""
        # Ensure the 'purchase_time' and 'signup_time' columns are datetime
        input_data['purchase_time'] = pd.to_datetime(input_data['purchase_time'], errors='coerce')
        input_data['signup_time'] = pd.to_datetime(input_data['signup_time'], errors='coerce')

        # Extract relevant time-based features
        input_data['purchase_day_of_week'] = input_data['purchase_time'].dt.dayofweek
        input_data['signup_day_of_week'] = input_data['signup_time'].dt.dayofweek
        input_data['signup_hour'] = input_data['signup_time'].dt.hour
        input_data['purchase__hour'] = input_data['purchase_time'].dt.hour

        # Drop original datetime columns
        input_data.drop(columns=['signup_time', 'purchase_time'], inplace=True)

        return input_data

    def encode_categorical_features(self, input_data):
        """Encodes categorical features, ensuring all expected columns are present."""
        categorical_features = ['source', 'browser', 'sex']

        # Convert categorical features to dummies with prefix
        encoded_data = pd.get_dummies(input_data, columns=categorical_features, prefix='category', drop_first=True)

        # Define all expected dummy columns based on model's training data
        expected_columns = [
            'category_Chrome', 'category_FireFox', 'category_IE', 'category_Opera', 
            'category_Safari', 'category_SEO', 'category_Ads', 'category_Direct',
            'category_F', 'category_M', 'category_Ads', 'category_Direct', 'category_SEO'
        ]

        # Add any missing columns as zero-filled to match model's training structure
        for col in expected_columns:
            if col not in encoded_data.columns:
                encoded_data[col] = 0

        # Ensure the columns order matches what the model was trained on
        encoded_data = encoded_data.reindex(columns=expected_columns, fill_value=0)

        return encoded_data
    def frequency_encoding(self, input_data):
        """Applies frequency encoding to the 'country' column."""
        # Check if 'country' column exists in the input_data
        if 'country' in input_data.columns:
            # Calculate the frequency of each category in the 'country' column
            frequency_map = input_data['country'].value_counts(normalize=True).to_dict()

            # Map the frequencies to the 'Country_encoded' column
            input_data['Country_encoded'] = input_data['country'].map(frequency_map)

            # Drop original 'country' column
            input_data.drop(columns=['country'], inplace=True)

        return input_data

    def normalize_and_scale(self, input_data):
        """Scales numerical features, ensuring columns exist before scaling."""
        numerical_features = [
            'purchase_value', 'age', 'purchase_day_of_week', 
            'signup_day_of_week', 'signup_hour', 'purchase__hour', 'Country_encoded'
        ]
        # Select features that are actually in the data to avoid KeyErrors
        features_to_scale = [feature for feature in numerical_features if feature in input_data.columns]
        
        # Scale only the columns present in the input data
        if features_to_scale:
            input_data[features_to_scale] = self.scaler.fit_transform(input_data[features_to_scale])
        
        return input_data


    def predict(self, input_data):
            """Main prediction method."""
            # Step 1: Add time-based features
            input_data = self.add_time_based_features(input_data)

            # Step 2: Encode categorical features
            input_data = self.encode_categorical_features(input_data)

            # Step 3: Apply frequency encoding
            input_data = self.frequency_encoding(input_data)

            # Step 4: Normalize and scale features
            input_data = self.normalize_and_scale(input_data)

            # Remove any duplicate columns to avoid reindexing issues
            input_data = input_data.loc[:, ~input_data.columns.duplicated()]

            # Step 5: Reorder columns to match model's expected order
            input_data = input_data.reindex(columns=self.feature_order, fill_value=0)

            # Make predictions
            prediction = self.model.predict(input_data)
          

            return prediction



fraud_model = FraudModel()  # Create an instance of your model

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        # Capture raw input data from the form
        data = request.json  
        
        # Ensure data is in the correct format for DataFrame
        if isinstance(data, dict):  # If a single record, wrap it in a list
            data = [data]
        
        # Convert JSON input to a DataFrame
        input_df = pd.DataFrame(data)  
        
        # Perform prediction
        predictions = fraud_model.predict(input_df)  # Call the model's predict method
        
        # Generate response message based on the prediction
        if predictions[0] == 0:
            result_message = "The User is Not Fraud Suspected"
        elif predictions[0] == 1:
            result_message = "The User is Fraud Suspected"
        else:
            result_message = "Unknown prediction result"

        # Prepare the response
        response = {"message": result_message}
        
        return jsonify(response)  # Return the response in JSON format
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error message if any exception occurs

if __name__ == '__main__':
    app.run(debug=True)
