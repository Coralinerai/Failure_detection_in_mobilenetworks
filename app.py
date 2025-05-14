import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load assets
@st.cache_data
def load_assets():
    with open('model/kmeans_model.pkl', 'rb') as f:
        assets = pickle.load(f)
    return assets['model'], assets['scaler']

kmeans, scaler = load_assets()

# App Title
st.title("📶 Mobile Network Failure Detector")
st.markdown("Predict failure states in mobile networks using K-Means clustering")

# Sidebar for input
st.sidebar.header("Input Parameters")
signal_strength = st.sidebar.slider("Signal Strength (dBm)", -120.0, -50.0, -85.0)
snr = st.sidebar.slider("SNR (dB)", 10.0, 30.0, 20.0)
attenuation = st.sidebar.slider("Attenuation", 0.0, 20.0, 5.0)
environment = st.sidebar.selectbox("Environment", ["Urban", "Home", "Open", "Suburban"])
call_duration = st.sidebar.slider("Call Duration (s)", 0, 2000, 500)
call_type = st.sidebar.selectbox("Call Type", ["voice", "data"])
distance_to_tower = st.sidebar.slider("Distance to Tower (km)", 0.0, 10.0, 1.5)
incoming_outgoing = st.sidebar.selectbox("Incoming/Outgoing", ["incoming", "outgoing"])

# Preprocess input
env_mapping = {"Urban": 0, "Home": 1, "Open": 2, "Suburban": 3}
call_type_mapping = {"voice": 0, "data": 1}
in_out_mapping = {"incoming": 0, "outgoing": 1}

# Calculate derived features (must match what was used in training)
signal_ratio = signal_strength / (snr + 1e-9)
atten_ratio = attenuation / (snr + 1e-9)

# Create input DataFrame with EXACTLY the same features as during training
input_data = pd.DataFrame([[
    signal_strength,                # Signal Strength (dBm)
    snr,                            # SNR
    attenuation,                    # Attenuation
    env_mapping[environment],       # Environment
    call_duration,                  # Call Duration (s)
    call_type_mapping[call_type],   # Call Type
    distance_to_tower,              # Distance to Tower (km)
    in_out_mapping[incoming_outgoing],  # Incoming/Outgoing
    signal_ratio,                   # Signal_SNR_Ratio
    atten_ratio                     # Attenuation_SNR_Ratio
]], columns=scaler.feature_names_in_)  # Critical: use scaler's feature names

# Scale and predict
scaled_input = scaler.transform(input_data)
cluster = kmeans.predict(scaled_input)[0]

# Display results
st.subheader("🔍 Prediction")
status = "🟢 Normal" if cluster == 0 else "🔴 Failure"
st.markdown(f"**Network Status:** {status} (Cluster {cluster})")

# Load and prepare sample data for visualization
@st.cache_data
def load_sample_data():
    data = pd.read_csv('data/cleaned_data.csv')
    data['Signal_SNR_Ratio'] = data['Signal Strength (dBm)'] / (data['SNR'] + 1e-9)
    data['Attenuation_SNR_Ratio'] = data['Attenuation'] / (data['SNR'] + 1e-9)
    
    # Remove unwanted columns (like Timestamp if it exists)
    if 'Timestamp' in data.columns:
        data = data.drop('Timestamp', axis=1)
    if 'Cluster' in data.columns:
        clusters = data['Cluster']
        features = data.drop('Cluster', axis=1)
    else:
        clusters = None
        features = data
        
    # Ensure we have all required features
    required_features = scaler.feature_names_in_
    missing_features = set(required_features) - set(features.columns)
    
    if missing_features:
        st.error(f"Missing features in training data: {missing_features}")
        return None, None
        
    # Reorder columns to match scaler's expectations
    features = features[required_features]
    
    return features, clusters

sample_features, sample_clusters = load_sample_data()

# PCA Visualization
if sample_features is not None:
    scaled_data = scaler.transform(sample_features)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots()
    
    if sample_clusters is not None:
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], 
                            c=sample_clusters, alpha=0.5)
        plt.legend(*scatter.legend_elements(), title="Clusters")
    
    ax.scatter(pca.transform(scaled_input)[:, 0], 
               pca.transform(scaled_input)[:, 1], 
               c='red', s=200, marker='X', label='Your Input')
    plt.title("PCA Projection of Network States")
    st.pyplot(fig)

# Cluster interpretation
st.subheader("📊 Cluster Analysis")
if cluster == 0:
    st.info("""
    **Normal Conditions**  
    - Avg Signal: -83.9 dBm  
    - Lower attenuation  
    - Stable SNR
    """)
else:
    st.error("""
    **Potential Failure**  
    - Avg Signal: -87.1 dBm  
    - High attenuation  
    - Unstable SNR
    """)