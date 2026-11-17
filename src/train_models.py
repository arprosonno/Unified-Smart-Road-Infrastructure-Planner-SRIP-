# src/train_models.py
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import os
from preprocessing import prepare_data

def train_all_models():
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    models = {}
    print("🚀 Training Smart Road Infrastructure Planning Models...\n")
    
    # -------------------------------------------------------------
    # Module 1: Road & Pavement Specifiers
    # -------------------------------------------------------------
    print("--> Training Module 1: Pavement Thickness (Regression)")
    thick_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    thick_model.fit(X_train, y_train['pavement_thickness_cm'])
    thick_preds = thick_model.predict(X_test)
    print(f"    MAE: {mean_absolute_error(y_test['pavement_thickness_cm'], thick_preds):.2f} cm")
    models['pavement_thickness'] = thick_model

    print("--> Training Module 1: Recommended Lanes (Classification)")
    lane_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lane_model.fit(X_train, y_train['recommended_lanes'])
    lane_preds = lane_model.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test['recommended_lanes'], lane_preds) * 100:.2f}%")
    models['recommended_lanes'] = lane_model

    print("--> Training Module 1: Pavement Grade (Classification)")
    grade_model = RandomForestClassifier(n_estimators=100, random_state=42)
    grade_model.fit(X_train, y_train['pavement_grade'])
    grade_preds = grade_model.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test['pavement_grade'], grade_preds) * 100:.2f}%")
    models['pavement_grade'] = grade_model

    # -------------------------------------------------------------
    # Module 2: Energy & Lighting Planner
    # -------------------------------------------------------------
    print("\n--> Training Module 2: Lighting Pole Spacing (Regression)")
    pole_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    pole_model.fit(X_train, y_train['pole_spacing_meters'])
    pole_preds = pole_model.predict(X_test)
    print(f"    MAE: {mean_absolute_error(y_test['pole_spacing_meters'], pole_preds):.2f} meters")
    models['pole_spacing'] = pole_model

    print("--> Training Module 2: Solar Hybrid Lighting Flag (Classification)")
    solar_model = XGBClassifier(n_estimators=100, random_state=42)
    solar_model.fit(X_train, y_train['solar_hybrid_flag'])
    solar_preds = solar_model.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test['solar_hybrid_flag'], solar_preds) * 100:.2f}%")
    models['solar_hybrid'] = solar_model

    # -------------------------------------------------------------
    # Module 3: Asset & Service Allocator
    # -------------------------------------------------------------
    print("\n--> Training Module 3: EV Charging Hubs Count (Regression)")
    ev_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ev_model.fit(X_train, y_train['ev_charging_hubs'])
    ev_preds = ev_model.predict(X_test)
    print(f"    MAE: {mean_absolute_error(y_test['ev_charging_hubs'], ev_preds):.2f} hubs")
    models['ev_hubs'] = ev_model

    print("--> Training Module 3: Rest/Service Stops Count (Regression)")
    rest_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rest_model.fit(X_train, y_train['rest_stops'])
    rest_preds = rest_model.predict(X_test)
    print(f"    MAE: {mean_absolute_error(y_test['rest_stops'], rest_preds):.2f} stops")
    models['rest_stops'] = rest_model

    # Save all models
    os.makedirs("models", exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, f"models/{name}.pkl")
    
    print("\n✅ All models trained successfully and saved to 'models/' directory.")

if __name__ == "__main__":
    train_all_models()