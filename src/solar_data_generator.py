# src/solar_data_generator.py
import numpy as np
import pandas as pd
import os

# Base monthly PSH factors (Jan=1.0, Mar=1.2, Jul=0.6, etc.)
MONTH_SOLAR_MULTIPLIERS = {
    "January": 0.95, "February": 1.10, "March": 1.25, "April": 1.20,
    "May": 1.05, "June": 0.70, "July": 0.65, "August": 0.75,
    "September": 0.85, "October": 1.05, "November": 1.10, "December": 0.95
}

DIVISIONS = {
    "Dhaka": {"base_psh": 4.5, "monsoon_days": 110},
    "Chittagong": {"base_psh": 4.8, "monsoon_days": 130},
    "Rajshahi": {"base_psh": 5.2, "monsoon_days": 85},
    "Sylhet": {"base_psh": 4.1, "monsoon_days": 150},
    "Khulna": {"base_psh": 4.7, "monsoon_days": 95},
    "Barishal": {"base_psh": 4.6, "monsoon_days": 115},
    "Rangpur": {"base_psh": 4.9, "monsoon_days": 90},
    "Mymensingh": {"base_psh": 4.3, "monsoon_days": 125}
}

def generate_solar_dataset(num_samples=3500, random_seed=42):
    np.random.seed(random_seed)
    
    cities = np.random.choice(list(DIVISIONS.keys()), size=num_samples)
    months = np.random.choice(list(MONTH_SOLAR_MULTIPLIERS.keys()), size=num_samples)
    
    psh_list = []
    for city, month in zip(cities, months):
        base_psh = DIVISIONS[city]["base_psh"]
        m_factor = MONTH_SOLAR_MULTIPLIERS[month]
        calc_psh = np.random.normal(base_psh * m_factor, 0.25)
        psh_list.append(max(1.2, calc_psh))
        
    panel_w = np.random.choice([60, 80, 100, 120, 150, 200], size=num_samples)
    battery_ah = np.random.choice([30, 40, 60, 80, 100, 150], size=num_samples)
    led_w = np.random.choice([30, 45, 60, 90, 120], size=num_samples)
    
    daily_gen_wh = panel_w * np.array(psh_list) * 0.75
    nightly_req_wh = led_w * 11.5 * 0.85
    battery_storage_wh = battery_ah * 12 * 0.8
    
    autonomy_days = battery_storage_wh / (nightly_req_wh + 1e-5)
    score = (daily_gen_wh / nightly_req_wh) + (autonomy_days * 0.35)
    
    solar_viable = (score + np.random.normal(0, 0.1, num_samples) > 1.30).astype(int)
    
    df = pd.DataFrame({
        'city': cities,
        'month': months,
        'peak_sun_hours': np.round(psh_list, 2),
        'panel_wattage_w': panel_w,
        'battery_capacity_ah': battery_ah,
        'led_power_w': led_w,
        'solar_viable': solar_viable
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True) # Ensure processed folder exists
    df = generate_solar_dataset()
    df.to_csv("data/raw/bd_solar_dataset.csv", index=False)
    print("✅ Monthly/City Solar dataset saved in 'data/raw/bd_solar_dataset.csv'.")