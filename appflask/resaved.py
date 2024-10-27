import joblib
model = joblib.load("../notebook/models/Fraud_Random_Forest.pkl")
joblib.dump(model, "Fraud_Random_Forest_compatible_v1_5_2.pkl")
