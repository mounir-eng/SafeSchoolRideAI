import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopy.distance
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer

# Configure app
st.set_page_config(page_title="School Bus AI", page_icon="🚌", layout="wide")

# App header
st.title("🚌 School Bus Safety & Maintenance Dashboard")
st.markdown("Integrated AI solution for driver behavior analysis and predictive maintenance")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload OBD-II Data (CSV)", type="csv")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

def process_data(df):
    """Initial data processing pipeline"""
    df.columns = df.columns.str.lower()
    df = df.bfill().ffill()
    
    # Feature engineering
    df['speed_prev'] = df['speed'].shift(1).fillna(method='bfill')
    df['rpm_prev'] = df['rpm'].shift(1).fillna(method='bfill')
    
    df['speed_increase'] = (
        (df['speed'] - df['speed_prev'])
        .clip(-50, 50)
        .fillna(0)
    )
    
    df['rpm_increase'] = (
        (df['rpm'] - df['rpm_prev'])
        .clip(-5000, 5000)
        .fillna(0)
    )
    
    # Dynamic thresholds
    df['dynamic_speed_limit'] = (
        df['speed']
        .rolling(100, min_periods=10)
        .quantile(0.85)
        .fillna(80)
    )
    
    df['idle_rpm_threshold'] = (
        df.loc[df['speed'] < 5, 'rpm']
        .rolling(100, min_periods=10)
        .quantile(0.90)
        .ffill()
        .fillna(1500)
    )
    
    return df

def calculate_scores(df):
    """Safety scoring system"""
    # Speeding score
    mask = df['speed'] > df['dynamic_speed_limit']
    df['speeding_score'] = (
        (-10 * (df['speed'] - df['dynamic_speed_limit']) / df['dynamic_speed_limit'])
        .where(mask, 0)
        .fillna(0)
    )
    
    # Idling score
    idle_mask = (df['speed'] < 5) & (df['rpm'] > df['idle_rpm_threshold'])
    df['idle_score'] = (
        (-5 * (df['rpm'] - df['idle_rpm_threshold']) / df['idle_rpm_threshold'])
        .where(idle_mask, 0)
        .fillna(0)
    )
    
    # Harsh braking
    df['harsh_braking_score'] = (
        (df['speed_prev'] - df['speed'])
        .apply(lambda x: -min(15, x//2.5) if x > 10 else 0)
        .fillna(0)
    )
    
    # Acceleration score
    accel_mask = (df['speed_increase'] > 15) & (df['rpm_increase'] > 750)
    df['acceleration_score'] = np.where(accel_mask, -10, 0)
    
    # Anomaly detection
    imputer = SimpleImputer(strategy='median')
    features = imputer.fit_transform(
        df[['speed', 'rpm', 'speed_increase', 'rpm_increase']]
    )
    clf = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly_score'] = pd.Series(
        np.where(clf.fit_predict(features) == -1, -20, 0),
        index=df.index
    )
    
    # Total score
    df["total_score"] = df[
        ['speeding_score', 'idle_score', 
         'harsh_braking_score', 'acceleration_score', 
         'anomaly_score']
    ].sum(axis=1).fillna(0)
    
    return df

def maintenance_system(df):
    """Predictive maintenance system"""
    # Engine health (0-100% scale)
    df['engine_stress'] = (
        ((df['rpm'] * df.get('engine_load', 50)) / 500000)
        .rolling(10, min_periods=1)
        .mean()
        .clip(0, 100)
        .fillna(0)
    )
    
    # Brake wear (0-100% scale)
    df['brake_wear'] = (
        (df['harsh_braking_score'].abs() / 15)
        .rolling(5, min_periods=1)
        .sum()
        .clip(0, 100)
        .fillna(0)
    )
    
    # Tire health simulation
    df['tire_wear'] = (
        (df['speed'].rolling(100).std() * 0.5)
        .clip(0, 100)
        .fillna(0)
    )
    
    # Maintenance prediction model
    X = df[['engine_stress', 'brake_wear', 'tire_wear']].fillna(0)
    y = np.random.randint(0, 2, size=len(df))  # Replace with real labels
    
    model = RandomForestClassifier()
    model.fit(X, y)
    df['maintenance_urgency'] = model.predict_proba(X)[:,1] * 100
    
    return df

def get_maintenance_recommendations(df):
    """Generate detailed maintenance actions"""
    last = df.iloc[-1]
    rec = {
        'status': 'green',
        'icon': '✅',
        'actions': []
    }
    
    # Status determination
    if last['maintenance_urgency'] > 75:
        rec.update({'status': 'red', 'icon': '🚨'})
    elif last['maintenance_urgency'] > 50:
        rec.update({'status': 'orange', 'icon': '⚠️'})
    
    # Component-specific recommendations
    components = {
        'engine_stress': ('Engine', 80, [
            "Check coolant levels",
            "Inspect oil quality",
            "Clean air filters"
        ]),
        'brake_wear': ('Brakes', 60, [
            "Measure pad thickness",
            "Check fluid levels",
            "Test ABS sensors"
        ]),
        'tire_wear': ('Tires', 70, [
            "Check tread depth",
            "Verify pressure",
            "Inspect for damage"
        ])
    }
    
    for metric, (name, threshold, steps) in components.items():
        if last[metric] > threshold:
            rec['actions'].append({
                'component': name,
                'urgency': f"{last[metric]:.1f}%",
                'steps': steps
            })
    
    return rec

# Main processing pipeline
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = process_data(df)
        df = calculate_scores(df)
        df = maintenance_system(df)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Data Processing Error: {str(e)}")
        st.stop()

# Dashboard layout
if st.session_state.df is not None:
    df = st.session_state.df
    maint_rec = get_maintenance_recommendations(df)
    
    # Header metrics
    cols = st.columns(4)
    cols[0].metric("Safety Score", f"{df['total_score'].sum():.0f}",
                  help="Cumulative safety score for the entire route")
    cols[1].metric("Maintenance Urgency", f"{df['maintenance_urgency'].iloc[-1]:.1f}%",
                  delta_color="inverse")
    cols[2].metric("Engine Stress", f"{df['engine_stress'].iloc[-1]:.1f}%")
    cols[3].metric("Brake Wear", f"{df['brake_wear'].iloc[-1]:.1f}%")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Safety", "Map", "Components", "Maintenance"])
    
    with tab1:
        # Safety timeline
        st.plotly_chart(px.line(
            df, x='timestamp', y='total_score',
            title="Safety Score Timeline",
            labels={'total_score': 'Safety Score', 'timestamp': 'Time'}
        ), use_container_width=True)
        
        # Violation breakdown
        cols = st.columns(2)
        cols[0].plotly_chart(px.pie(
            names=['Speeding', 'Braking', 'Acceleration'],
            values=[
                df['speeding_score'].abs().sum(),
                df['harsh_braking_score'].abs().sum(),
                df['acceleration_score'].abs().sum()
            ],
            title="Violation Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        ), use_container_width=True)
        
        # Score distribution
        cols[1].plotly_chart(px.histogram(
            df, x='total_score', 
            title="Score Distribution",
            nbins=20,
            color_discrete_sequence=['#4CAF50']
        ), use_container_width=True)
    
    with tab2:
        if 'lat' in df.columns and 'lon' in df.columns:
            st.plotly_chart(px.scatter_mapbox(
                df, lat='lat', lon='lon', color='total_score',
                mapbox_style="open-street-map", zoom=12,
                title="Route Safety Analysis",
                color_continuous_scale=px.colors.cyclical.IceFire
            ), use_container_width=True)
        else:
            st.warning("No geospatial data available")
    
    with tab3:
        st.subheader("Component Health Analysis")
        
        # Real-time gauges
        gauge_cols = st.columns(3)
        components = [
            ('engine_stress', 'Engine Stress', '#FFA500'),
            ('brake_wear', 'Brake Wear', '#FF4444'),
            ('tire_wear', 'Tire Wear', '#4CAF50')
        ]
        
        for idx, (metric, title, color) in enumerate(components):
            with gauge_cols[idx]:
                fig = px.bar(
                    x=[title],
                    y=[df[metric].iloc[-1]],
                    range_y=[0, 100],
                    title=title,
                    color_discrete_sequence=[color],
                    text=[f"{df[metric].iloc[-1]:.1f}%"]
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        # Health trends
        st.plotly_chart(px.line(
            df, x='timestamp', y=['engine_stress', 'brake_wear', 'tire_wear'],
            title="Component Health History",
            labels={'value': 'Health Status (%)'},
            color_discrete_map={
                'engine_stress': '#FFA500',
                'brake_wear': '#FF4444',
                'tire_wear': '#4CAF50'
            }
        ), use_container_width=True)
    
    with tab4:
        st.subheader(f"Maintenance Recommendations {maint_rec['icon']}")
        
        # Status header
        st.markdown(f"""
        <div style="padding:20px; border-radius:10px; 
                    background-color: {'#FFEBEE' if maint_rec['status'] == 'red' else 
                                      '#FFF3E0' if maint_rec['status'] == 'orange' else 
                                      '#E8F5E9'};">
            <h3 style="color: {'#D32F2F' if maint_rec['status'] == 'red' else 
                             '#EF6C00' if maint_rec['status'] == 'orange' else 
                             '#2E7D32'};">{maint_rec['status'].upper()} STATUS</h3>
            <p>{len(maint_rec['actions'])} critical maintenance actions required</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Maintenance actions
        if maint_rec['actions']:
            for action in maint_rec['actions']:
                with st.expander(f"{action['component']} ({action['urgency']})", expanded=True):
                    st.markdown("**Required Steps:**")
                    for step in action['steps']:
                        st.markdown(f"- {step}")
                    st.button("Mark as Completed", key=f"btn_{action['component']}")
        else:
            st.success("No critical maintenance actions required")
        
        # Maintenance schedule
        st.subheader("Preventive Maintenance Schedule")
        st.markdown("""
        | Component | Frequency | Next Due       |
        |-----------|-----------|----------------|
        | Engine    | 5,000 km  | 1,234 km       |
        | Brakes    | 10,000 km | 5,678 km       |
        | Tires     | 15,000 km | 9,012 km       |
        """)

else:
    st.info("👈 Upload OBD-II CSV data to begin analysis")

# Style enhancements
st.markdown("""
<style>
    [data-testid=stMetric] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .stPlotlyChart {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 10px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }
    th, td {
        border: 1px solid #e0e0e0;
        padding: 8px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)