# src/data_generator.py
import numpy as np
import pandas as pd
import os

def generate_route_dataset(num_samples=3500, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Base Input Route Features
    daily_traffic_volume = np.random.exponential(scale=25000, size=num_samples) + 2000
    daily_traffic_volume = np.clip(daily_traffic_volume, 1000, 95000)
    
    heavy_truck_ratio = np.random.beta(a=2, b=5, size=num_samples) * 0.45 + 0.02
    route_length_km = np.random.uniform(5.0, 180.0, size=num_samples)
    urban_density_index = np.random.uniform(0.05, 0.98, size=num_samples)
    ev_adoption_rate = np.random.uniform(0.02, 0.45, size=num_samples)
    night_traffic_ratio = np.random.uniform(0.08, 0.42, size=num_samples)
    grid_proximity_km = np.random.exponential(scale=8.0, size=num_samples) + 0.5
    
    # 2. Targets with Calibrated Engineering Noise
    
    # Pavement Thickness (cm)
    heavy_truck_count = daily_traffic_volume * heavy_truck_ratio
    thickness_base = 10.0 + (heavy_truck_count ** 0.65) * 0.05 + (daily_traffic_volume * 0.00008)
    pavement_thickness_cm = np.clip(thickness_base + np.random.normal(0, 1.8, num_samples), 8.0, 48.0)
    
    # Recommended Lanes (Tuned boundary noise)
    lane_score = (daily_traffic_volume / 12000.0) + (urban_density_index * 1.2) + np.random.normal(0, 0.25, num_samples)
    lane_count = np.where(lane_score < 1.8, 2,
                 np.where(lane_score < 3.8, 4,
                 np.where(lane_score < 5.8, 6, 8)))
    
    # Pavement Grade (Tuned boundary noise)
    grade_score = (heavy_truck_ratio * 3.8) + (daily_traffic_volume / 35000.0) + np.random.normal(0, 0.18, num_samples)
    pavement_grade = np.where(grade_score < 0.85, 0,
                      np.where(grade_score < 1.65, 1, 2))
    
    # Lighting Pole Spacing (meters)
    spacing_base = 42.0 - (urban_density_index * 14.0) - (night_traffic_ratio * 18.0)
    pole_spacing_meters = np.clip(spacing_base + np.random.normal(0, 1.2, num_samples), 18.0, 50.0)
    
    # Solar Hybrid Flag
    solar_prob = 1 / (1 + np.exp(-(grid_proximity_km - 8.5) / 1.5))
    solar_hybrid_flag = (np.random.uniform(0, 1, num_samples) < solar_prob).astype(int)
    
    # EV Charging Hubs & Rest Stops
    ev_hubs_raw = (route_length_km / 18.0) * (ev_adoption_rate * 2.2) + np.random.normal(0, 0.4, num_samples)
    ev_charging_hubs = np.clip(np.round(ev_hubs_raw), 0, 22).astype(int)
    
    rest_stops_raw = (route_length_km / 35.0) * (0.8 + heavy_truck_ratio * 1.2) + np.random.normal(0, 0.3, num_samples)
    rest_stops = np.clip(np.round(rest_stops_raw), 0, 12).astype(int)
    
    df = pd.DataFrame({
        'daily_traffic_volume': np.round(daily_traffic_volume, 0),
        'heavy_truck_ratio': np.round(heavy_truck_ratio, 4),
        'route_length_km': np.round(route_length_km, 2),
        'urban_density_index': np.round(urban_density_index, 3),
        'ev_adoption_rate': np.round(ev_adoption_rate, 3),
        'night_traffic_ratio': np.round(night_traffic_ratio, 3),
        'grid_proximity_km': np.round(grid_proximity_km, 2),
        'pavement_thickness_cm': np.round(pavement_thickness_cm, 2),
        'recommended_lanes': lane_count,
        'pavement_grade': pavement_grade,
        'pole_spacing_meters': np.round(pole_spacing_meters, 1),
        'solar_hybrid_flag': solar_hybrid_flag,
        'ev_charging_hubs': ev_charging_hubs,
        'rest_stops': rest_stops
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = generate_route_dataset()
    output_path = "data/raw/route_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Generated calibrated dataset with {len(df)} rows saved to '{output_path}'.")