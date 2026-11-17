# src/train_solar_model.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Ensure data folders exist
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/bd_solar_dataset.csv")

X = df[['city', 'month', 'peak_sun_hours', 'panel_wattage_w', 'battery_capacity_ah', 'led_power_w']]
y = df['solar_viable']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'month'])
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_trans, y_train)

preds = clf.predict(X_test_trans)
acc = accuracy_score(y_test, preds)

print(f"☀️ Monthly Solar Feasibility Model Trained! Accuracy: {acc * 100:.2f}%")

# Save processed sample & models
pd.DataFrame(X_train_trans.toarray() if hasattr(X_train_trans, 'toarray') else X_train_trans).to_csv("data/processed/solar_processed_features.csv", index=False)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/solar_feasibility_model.pkl")
joblib.dump(preprocessor, "models/solar_preprocessor.pkl")
print("✅ Saved models to 'models/' and processed data to 'data/processed/'.")