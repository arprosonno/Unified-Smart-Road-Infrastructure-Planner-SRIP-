# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    'daily_traffic_volume',
    'heavy_truck_ratio',
    'route_length_km',
    'urban_density_index',
    'ev_adoption_rate',
    'night_traffic_ratio',
    'grid_proximity_km'
]

def prepare_data(data_path="data/raw/route_dataset.csv"):
    df = pd.read_csv(data_path)
    
    X = df[FEATURE_COLS]
    y = df.drop(columns=FEATURE_COLS)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler