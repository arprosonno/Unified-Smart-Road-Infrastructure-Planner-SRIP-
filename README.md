# 🛣️ Unified Smart Road Infrastructure Planner (SRIP)

An AI-driven civil engineering decision engine that translates raw route traffic metrics, demographics, and geographical factors into an automated, comprehensive road construction and infrastructure blueprint.

---

## 📌 Project Overview

Urban infrastructure planning often relies on static manual guidelines rather than dynamic, data-driven insights. **SRIP** solves this by unifying multiple road planning sub-domains into a single machine learning pipeline. 

Given basic route profile inputs (traffic volume, heavy truck ratios, urban density, EV adoption), the system predicts and allocates:
1. **Road & Pavement Specs:** Recommended lane configurations, asphalt thickness (cm), and structural material grades.
2. **Energy & Lighting:** Streetlight pole spacing intervals, total pole requirements, and grid vs. solar-hybrid power specs.
3. **Service & Commercial Assets:** Optimal placement density for EV fast-charging hubs and highway rest stops.

---

## 📂 Project Architecture & Directory Structure

```text
smart_road_planner/
│
├── data/
│   ├── raw/                  # Generated synthetic route dataset (route_dataset.csv)
│   └── processed/            # Processed / scaled datasets
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_generator.py     # Physics-correlated synthetic data generation script
│   ├── preprocessing.py      # Feature scaling & train/test splitting logic
│   ├── train_models.py       # Training pipeline for multi-module ML models
│   └── inference.py          # Command-line blueprint generation wrapper
│
├── models/                   # Saved Scikit-Learn & XGBoost artifacts (.pkl)
│   ├── scaler.pkl
│   ├── pavement_thickness.pkl
│   ├── recommended_lanes.pkl
│   ├── pavement_grade.pkl
│   ├── pole_spacing.pkl
│   ├── solar_hybrid.pkl
│   ├── ev_hubs.pkl
│   └── rest_stops.pkl
│
├── notebooks/                # Jupyter notebooks for EDA and testing
│   └── EDA_and_Testing.ipynb
│
├── app.py                    # Streamlit Web UI Dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation