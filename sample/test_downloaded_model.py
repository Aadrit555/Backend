import pandas as pd
from autogluon.tabular import TabularPredictor

model_path = "/home/ojasvkushwah/Downloads/model_de619009"

try:
    print(f"Loading model from {model_path}...")
    predictor = TabularPredictor.load(model_path)
    
    print("\n--- Model Information ---")
    features = predictor.feature_metadata_in.get_features()
    print("Expected Features:", features)
    print("Target Column:", predictor.label)
    
    # Create a dummy row for prediction based on the features
    dummy_data = {}
    for feature in features:
        # Just put a 0 or 'unknown' for dummy inference depending on type, but for now we'll just put 0
        # A better way is to check the types, but this is just a quick inspection.
        dummy_data[feature] = 0
        
    df = pd.DataFrame([dummy_data])
    print("\nPredicting on dummy data:", dummy_data)
    
    prediction = predictor.predict(df)
    print("\nPrediction Result:")
    print(prediction.iloc[0])
    
except Exception as e:
    print(f"Error: {e}")
