

📡 Failure Detection in Mobile Networks

A machine learning approach to proactively detect network failures using clustering techniques.

This project was developed in collaboration with:

Boudalal Oumaima @Coralinerai
Antar Wiam  @Wiamant

📌 Project Overview
Goal: Develop a predictive system to detect failures in mobile networks using unsupervised learning (K-Means clustering) to classify network states as "failure" or "normal."

Key Features:

Data preprocessing (outlier handling, categorical encoding, feature engineering).

Exploratory Data Analysis (EDA) to identify signal strength patterns.

K-Means clustering to segment network performance data.

Evaluation using Silhouette Score (0.40) and Davies-Bouldin Index (1.02).

📂 Dataset
Size: 463 rows × 10 columns.

Features:

Numerical: Signal Strength (dBm), SNR, Call Duration, Attenuation, Distance to Tower.

Categorical: Environment (urban, home, etc.), Call Type (voice/data), Incoming/Outgoing.

Preprocessing:

Outliers replaced using IQR method.

Categorical variables encoded numerically.

Time-series data standardized.

🔍 Exploratory Analysis (EDA)
Signal Strength: Varied significantly by environment (e.g., weaker in urban areas).

SNR: Limited correlation with signal strength.

Attenuation: More dependent on physical distance than environment.

Visualizations: Boxplots, scatter plots, and correlation matrices (see code).

🤖 Model Building

K-Means Clustering
Features: Signal_SNR_Ratio, Attenuation_SNR_Ratio (engineered).

Optimal Clusters: 2 (determined by the Elbow Method).

Results:

Cluster 0: Normal conditions (higher signal strength, lower attenuation).

Cluster 1: Failure/degraded conditions (weaker signal, higher attenuation).

Evaluation:


Silhouette Score: 0.40  # Moderate separation
Davies-Bouldin Index: 1.02  # Low = better distinction

🚧 Limitations & Future Work

Challenges: Weak feature correlations, small dataset.

Improvements:

Collect richer data (e.g., latency, packet loss metrics).

Test supervised models if labeled data becomes available.

Deploy in real-time network monitoring systems.

🛠️ Installation & Usage

Clone the repo:

git clone https://github.com/Coralinerai/Failure_detection_in_mobilenetworks.git



Dependencies:

python 
pandas, numpy, scikit-learn, matplotlib, seaborn

📜 License
MIT




