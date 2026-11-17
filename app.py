# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Smart Road Infrastructure Planner (SRIP)",
    page_icon="🛣️",
    layout="wide"
)

# Regional Environmental Metadata
MONTH_FACTORS = {
    "January": 0.95, "February": 1.10, "March": 1.25, "April": 1.20,
    "May": 1.05, "June": 0.70, "July": 0.65, "August": 0.75,
    "September": 0.85, "October": 1.05, "November": 1.10, "December": 0.95
}

CITIES = {
    "Dhaka": 4.5, "Chittagong": 4.8, "Rajshahi": 5.2, "Sylhet": 4.1,
    "Khulna": 4.7, "Barishal": 4.6, "Rangpur": 4.9, "Mymensingh": 4.3
}

# Header Section
st.title("🛣️ Unified Smart Road Infrastructure Planner (SRIP)")
st.markdown("""
*An AI-driven decision engine translating route demographics into multi-domain civil blueprints and localized monthly solar viability assessments.*
""")
st.divider()

# Session State Initialization to prevent resets on widget change
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Layout Columns
col_inputs, col_outputs = st.columns([1, 2], gap="large")

with col_inputs:
    st.header("🎛️ Input Demographics")
    
    traffic_vol = st.number_input("Daily Traffic Volume (vehicles/day)", 1000, 200000, 50000, step=1000)
    truck_ratio = st.slider("Heavy Truck Share (%)", 0.0, 100.0, 45.0) / 100.0
    length_km = st.slider("Route Length (km)", 1.0, 10000.0, 180.0, step=5.0)
    urban_density = st.slider("Urban Density Index (0=Rural, 1=Urban)", 0.0, 1.0, 0.85)
    ev_rate = st.slider("EV Adoption Rate (%)", 0.0, 100.0, 50.0) / 100.0
    night_ratio = st.slider("Night Traffic Share (%)", 0.0, 100.0, 42.0) / 100.0
    grid_dist = st.slider("Distance to Power Substation (km)", 0.1, 10000.0, 25.0)
    
    if st.button("🚀 Generate Infrastructure Blueprint", use_container_width=True, type="primary"):
        st.session_state.predicted = True

with col_outputs:
    st.header("🏗️ Construction Blueprint")
    
    if st.session_state.predicted:
        try:
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
            
            input_df = pd.DataFrame([{
                'daily_traffic_volume': traffic_vol,
                'heavy_truck_ratio': truck_ratio,
                'route_length_km': length_km,
                'urban_density_index': urban_density,
                'ev_adoption_rate': ev_rate,
                'night_traffic_ratio': night_ratio,
                'grid_proximity_km': grid_dist
            }])
            
            scaled = scaler.transform(input_df)
            
            thick = float(models['pavement_thickness'].predict(scaled)[0])
            lanes = int(models['recommended_lanes'].predict(scaled)[0])
            grade = int(models['pavement_grade'].predict(scaled)[0])
            spacing = float(models['pole_spacing'].predict(scaled)[0])
            solar = int(models['solar_hybrid'].predict(scaled)[0])
            ev = int(np.round(models['ev_hubs'].predict(scaled)[0]))
            rest = int(np.round(models['rest_stops'].predict(scaled)[0]))
            
            grade_map = {
                0: "Standard Bitumen Pavement (Light Duty)",
                1: "Polymer-Modified Asphalt (Medium-Heavy Load)",
                2: "Reinforced Heavy-Duty Concrete (Freight Corridor)"
            }
            
            total_poles = int(np.ceil((length_km * 1000) / max(spacing, 1)))
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🛣️ Pavement & Lanes", 
                "💡 Energy & Lighting", 
                "⚡ Services & Assets", 
                "📊 Route Analytics"
            ])
            
            # ---------------- TAB 1: CIVIL ----------------
            with tab1:
                st.subheader("Module 1: Civil & Structural Road Specs")
                m1, m2, m3 = st.columns(3)
                m1.metric("Recommended Lanes", f"{lanes} Lanes", f"{lanes//2} per direction")
                m2.metric("Asphalt Thickness", f"{thick:.2f} cm")
                m3.metric("Material Grade", f"Tier {grade}")
                
                st.info(f"**Structural Recommendation:** {grade_map[grade]}")
                
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = thick,
                    title = {'text': "Asphalt Structural Depth Load Gauge (cm)"},
                    gauge = {
                        'axis': {'range': [5, 50]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [5, 20], 'color': "#e0f2fe"},
                            {'range': [20, 35], 'color': "#bae6fd"},
                            {'range': [35, 50], 'color': "#0284c7"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # ---------------- TAB 2: LIGHTING & MONTHLY SOLAR ----------------
            with tab2:
                st.subheader("Module 2: Streetlighting & Monthly Solar Viability")
                l1, l2, l3 = st.columns(3)
                l1.metric("Pole Spacing", f"{spacing:.1f} meters")
                l2.metric("Total Poles Required", f"{total_poles:,} units")
                l3.metric("Grid Source Recommendation", "Solar-Hybrid" if solar == 1 else "Grid Connection")
                
                st.divider()
                st.markdown("### 🇧🇩 Regional Monthly Solar Engine")
                
                col_bd1, col_bd2 = st.columns(2)
                
                with col_bd1:
                    selected_city = st.selectbox("Select City / Division", list(CITIES.keys()), key="city_selector")
                    selected_month = st.selectbox("Select Month", list(MONTH_FACTORS.keys()), index=2, key="month_selector")
                    panel_w = st.select_slider("Solar Panel Rating per Pole (Watts)", options=[60, 80, 100, 120, 150, 200], value=100)
                    battery_ah = st.select_slider("Battery Capacity (Ah @ 12V)", options=[30, 40, 60, 80, 100, 150], value=60)
                    led_w = st.select_slider("LED Fixture Power (Watts)", options=[30, 45, 60, 90, 120], value=45)

                try:
                    solar_clf = joblib.load("models/solar_feasibility_model.pkl")
                    solar_prep = joblib.load("models/solar_preprocessor.pkl")
                    
                    psh_est = round(CITIES[selected_city] * MONTH_FACTORS[selected_month], 2)
                    
                    input_solar_df = pd.DataFrame([{
                        'city': selected_city,
                        'month': selected_month,
                        'peak_sun_hours': psh_est,
                        'panel_wattage_w': panel_w,
                        'battery_capacity_ah': battery_ah,
                        'led_power_w': led_w
                    }])
                    
                    trans_input = solar_prep.transform(input_solar_df)
                    is_viable = solar_clf.predict(trans_input)[0]
                    
                    with col_bd2:
                        st.write("#### Monthly Feasibility Analysis")
                        st.write(f"**City:** {selected_city}")
                        st.write(f"**Month:** {selected_month}")
                        st.write(f"**Estimated Peak Sun Hours:** ~{psh_est} hrs/day")
                        
                        if is_viable == 1:
                            st.success(f"🟢 **Off-Grid Solar Viable in {selected_month}:** The {panel_w}W panel + {battery_ah}Ah battery generates sufficient solar power in {selected_city} for {selected_month}.")
                        else:
                            st.error(f"🔴 **Grid-Hybrid Backup Needed in {selected_month}:** Low solar irradiance in {selected_city} during {selected_month}. Increase panel/battery capacity or rely on grid tie-in.")

                except Exception as e:
                    st.info("💡 Run `python src/train_solar_model.py` to enable the monthly solar engine.")

            # ---------------- TAB 3: SERVICES ----------------
            with tab3:
                st.subheader("Module 3: Commercial & Service Stations")
                s1, s2 = st.columns(2)
                s1.metric("EV Fast-Charging Hubs", f"{ev} Hubs", f"Every ~{length_km/max(ev, 1):.1f} km")
                s2.metric("Rest & Fuel Stations", f"{rest} Stations", f"Every ~{length_km/max(rest, 1):.1f} km")

            # ---------------- TAB 4: ANALYTICS ----------------
            with tab4:
                st.subheader("Traffic Breakdown Analysis")
                heavy_vol = traffic_vol * truck_ratio
                light_vol = traffic_vol - (traffic_vol * min(truck_ratio, 1.0))
                
                fig_pie = px.pie(
                    names=['Passenger Vehicles / Bikes', 'Heavy Freight Trucks / Buses'],
                    values=[max(0, light_vol), heavy_vol],
                    title="Daily Traffic Composition Split",
                    color_discrete_sequence=['#3b82f6', '#ef4444']
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as e:
            st.error(f"Error executing prediction. Ensure models are trained! ({e})")
    else:
        st.info("👈 Adjust parameters on the left and click **Generate Infrastructure Blueprint** to view the AI output.")