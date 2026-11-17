# src/inference.py
import numpy as np
import pandas as pd
import joblib

GRADE_MAP = {
    0: "Standard Bitumen Pavement (Light Traffic)",
    1: "Polymer-Modified Asphalt (Medium-Heavy Load)",
    2: "Reinforced Heavy-Duty Concrete (Heavy Freight Corridor)"
}

def generate_road_blueprint(route_data_dict):
    """
    Accepts a dictionary of input route parameters and prints a structured engineering blueprint.
    """
    scaler = joblib.load("models/scaler.pkl")
    
    models = {
        'pavement_thickness': joblib.load("models/pavement_thickness.pkl"),
        'recommended_lanes': joblib.load("models/recommended_lanes.pkl"),
        'pavement_grade': joblib.load("models/pavement_grade.pkl"),
        'pole_spacing': joblib.load("models/pole_spacing.pkl"),
        'solar_hybrid': joblib.load("models/solar_hybrid.pkl"),
        'ev_hubs': joblib.load("models/ev_hubs.pkl"),
        'rest_stops': joblib.load("models/rest_stops.pkl")
    }
    
    input_df = pd.DataFrame([route_data_dict])
    scaled_input = scaler.transform(input_df)
    
    # Predict
    p_thickness = float(models['pavement_thickness'].predict(scaled_input)[0])
    p_lanes = int(models['recommended_lanes'].predict(scaled_input)[0])
    p_grade_id = int(models['pavement_grade'].predict(scaled_input)[0])
    p_spacing = float(models['pole_spacing'].predict(scaled_input)[0])
    p_solar = int(models['solar_hybrid'].predict(scaled_input)[0])
    p_ev = int(np.round(models['ev_hubs'].predict(scaled_input)[0]))
    p_rest = int(np.round(models['rest_stops'].predict(scaled_input)[0]))
    
    # Total streetlights estimated
    total_poles = int(np.ceil((route_data_dict['route_length_km'] * 1000) / p_spacing))
    
    # Print Executive Report
    print("\n" + "="*70)
    print("      UNIFIED SMART ROAD INFRASTRUCTURE PLANNER (SRIP)")
    print("                 AUTOMATED CONSTRUCTION BLUEPRINT")
    print("="*70)
    print(f"📌 INPUT ROUTE PROFILE:")
    print(f"   • Route Distance:           {route_data_dict['route_length_km']} km")
    print(f"   • Daily Traffic Estimate:   {route_data_dict['daily_traffic_volume']:,} vehicles/day")
    print(f"   • Heavy Truck Ratio:        {route_data_dict['heavy_truck_ratio']*100:.1f}%")
    print(f"   • Urban Density Score:      {route_data_dict['urban_density_index']:.2f} (0=Rural, 1=Urban)")
    print(f"   • EV Adoption Factor:       {route_data_dict['ev_adoption_rate']*100:.1f}%")
    print("-" * 70)
    print("🏗️  MODULE 1: ROAD & PAVEMENT SPECIFICATIONS")
    print(f"   • Recommended Lanes:        {p_lanes} Lanes ({p_lanes//2} in each direction)")
    print(f"   • Asphalt Thickness:        {p_thickness:.2f} cm")
    print(f"   • Structural Material:      {GRADE_MAP[p_grade_id]}")
    print("-" * 70)
    print("💡 MODULE 2: ENERGY & STREETLIGHTING PLAN")
    print(f"   • Pole Spacing Interval:    Every {p_spacing:.1f} meters")
    print(f"   • Total Lighting Poles:     ~{total_poles:,} units across route")
    print(f"   • Recommended Power Spec:   {'Solar-Grid Hybrid (Off-Grid Resilient)' if p_solar == 1 else 'Main Grid Connected'}")
    print("-" * 70)
    print("⛽ MODULE 3: COMMERCIAL & SERVICE ASSET ALLOCATION")
    print(f"   • EV Fast-Charging Hubs:    {p_ev} locations along corridor")
    print(f"   • Highway Rest/Fuel Stops:   {p_rest} service stations")
    print("="*70 + "\n")

if __name__ == "__main__":
    # Test Sample Route (e.g., A busy 45km highway corridor)
    sample_route = {
        'daily_traffic_volume': 38000,
        'heavy_truck_ratio': 0.28,
        'route_length_km': 45.0,
        'urban_density_index': 0.65,
        'ev_adoption_rate': 0.25,
        'night_traffic_ratio': 0.20,
        'grid_proximity_km': 12.5
    }
    generate_road_blueprint(sample_route)