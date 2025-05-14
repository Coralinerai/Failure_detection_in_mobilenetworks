import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import missingno as msno 
import datetime as dt
data = pd.read_csv('data/cleaned_data.csv')
#Etape 1 : Visualisation univarié
#colonnes numériques 
#Statistiques descriptives
for column in data.select_dtypes(include=['number']).columns:
    print(f"Statistiques pour {column} :")
    print(data[column].describe())
#Distribution (Histogramme)
for column in data.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f"Distribution de {column}")
    plt.show()
#Boîte à moustaches (Boxplot)
for column in data.select_dtypes(include=['number']).columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data[column])
    plt.title(f"Boxplot de {column}")
    plt.show()
# Liste des colonnes catégoriques (après transformation)
categorical_columns = ['Environment', 'Call Type', 'Incoming/Outgoing']
for col in categorical_columns:
    # Calcul des fréquences
    counts = data[col].value_counts()
    percentages = counts / counts.sum() * 100  # Calcul des pourcentages
    
    # Création de la figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Diagramme à barres
    axes[0].bar(counts.index, counts.values, color='skyblue')
    axes[0].set_title(f'Bar Chart for {col}')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xticks(counts.index)
    axes[0].set_xticklabels(counts.index, rotation=0)  # Labels propres
    
    # Diagramme circulaire
    axes[1].pie(percentages, labels=counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    axes[1].set_title(f'Pie Chart for {col}')
    
    # Affichage
    plt.tight_layout()
    plt.show()


#Etape 2:Relation entre colonnes 
#correlation entre signal strength et SNR 
correlation=data['Signal Strength (dBm)'].corr(data['SNR'])
print("Correlation between Signal Strength and SNR :", correlation)
#Correlation entre Signal strength et attnuation 
correlation=data['Signal Strength (dBm)'].corr(data['Attenuation'])
print("Correlation between Signal Strength and Attenuation :", correlation)
#Correlation entre SNR et attenuation
correlation=data['Attenuation'].corr(data['SNR'])
print("Correlation between Attenuation and SNR :", correlation)

#Correlation entre SNR et attenuation
correlation=data['Attenuation'].corr(data['Distance to Tower (km)'])
print("Correlation between Attenuation and Distance to Tower (km) :", correlation)

#visual analysis
plt.scatter(data['Signal Strength (dBm)'],data['SNR'])
plt.title('Signal Strength vs SNR')
plt.xlabel('Signal Strength (dBm)')
plt.ylabel('SNR')
plt.show()
plt.scatter(data['Attenuation'],data['SNR'])
plt.title('Attenuation vs SNR')
plt.xlabel('Attenuation')
plt.ylabel('SNR')
plt.show()
plt.scatter(data['Signal Strength (dBm)'],data['Attenuation'])
plt.title('Signal Strength vs Attenuation')
plt.xlabel('Signal Strength (dBm)')
plt.ylabel('Attenuation')
plt.show()
plt.scatter(data['Distance to Tower (km)'],data['Attenuation'])
plt.title('Distance to Tower (km) vs Attenuation')
plt.xlabel('Distance to Tower (km)')
plt.ylabel('Attenuation')
plt.show()


#Inspecting data 
#  Checking Signal Strength range
print(data['Signal Strength (dBm)'].min(), data['Signal Strength (dBm)'].max())
# Low variance detection
low_variance_cols = data[['Attenuation', 'Distance to Tower (km)', 'Signal Strength (dBm)', 'SNR']].var()
low_variance_cols = low_variance_cols[low_variance_cols < 0.01]
print("Low variance columns:")
print(low_variance_cols)
# Correlation heatmap
sns.heatmap(data[['Attenuation', 'Distance to Tower (km)', 'Signal Strength (dBm)', 'SNR']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
#Check for negative distances
invalid_distances = data[data['Distance to Tower (km)'] < 0]
print("Rows with invalid distances:")
print(invalid_distances)
# Visualize distributions
data[['Attenuation', 'Distance to Tower (km)', 'Signal Strength (dBm)', 'SNR']].hist(bins=20, figsize=(10, 8))
plt.show()
# Boxplot for Attenuation and Signal Strength across different environments
plt.figure(figsize=(14, 6))
# Subplot for Attenuation
plt.subplot(1, 2, 1)
sns.boxplot(x='Environment', y='Attenuation', data=data)
plt.title('Attenuation by Environment')
plt.xticks(rotation=45)
# Subplot for Signal Strength
plt.subplot(1, 2, 2)
sns.boxplot(x='Environment', y='Signal Strength (dBm)', data=data)
plt.title('Signal Strength by Environment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#scatterplot to show the relationship
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Attenuation', y='Signal Strength (dBm)', hue='Environment', data=data)
plt.title('Signal Strength vs. Attenuation by Environment')
plt.legend(title='Environment')
plt.show()
# Boxplot for Attenuation and Signal Strength across different environments
plt.figure(figsize=(14, 6))
# Subplot for Attenuation
plt.subplot(1, 2, 1)
sns.boxplot(x='Call Type', y='Attenuation', data=data)
plt.title('Attenuation by Environment')
plt.xticks(rotation=45)
# Subplot for Signal Strength
plt.subplot(1, 2, 2)
sns.boxplot(x='Call Type', y='Signal Strength (dBm)', data=data)
plt.title('Signal Strength by Environment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# scatterplot to show the relationship
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Attenuation', y='Signal Strength (dBm)', hue='Call Type', data=data)
plt.title('Signal Strength vs. Attenuation by Call Type')
plt.legend(title='Call Type')
plt.show()
grouped = data.groupby('Environment')
# Create a dictionary of subsets for dynamic access
environment_subsets = {env: group for env, group in grouped}
# Step 3: Analyze each subset
for env, subset in environment_subsets.items():
    mean_strength = subset['Signal Strength (dBm)'].mean()
    print(f"Mean Signal Strength (Environment {env}): {mean_strength}")
# Step 4: Plot Signal Strength vs. Attenuation for all environments
colors = ['#FF9999', '#66B2FF', '#99FF99', '#CC99FF']  # Define colors for each environment
for env, subset in environment_subsets.items():
    plt.scatter(
        subset['Attenuation'], 
        subset['Signal Strength (dBm)'], 
        label=f"Environment {env}", 
        alpha=0.6, 
        color=colors[env]
    )
plt.xlabel("Attenuation")
plt.ylabel("Signal Strength (dBm)")
plt.title("Signal Strength vs. Attenuation (by Environment)")
plt.legend()
plt.show()
grouped = data.groupby('Incoming/Outgoing')
environment_subsets = {env: group for env, group in grouped}
for env, subset in environment_subsets.items():
    mean_strength = subset['Signal Strength (dBm)'].mean()
    print(f"Mean Signal Strength (Incoming/Outgoing {env}): {mean_strength}")
colors = ['#FF9999', '#66B2FF']  
for env, subset in environment_subsets.items():
    plt.scatter(
        subset['Attenuation'], 
        subset['Signal Strength (dBm)'], 
        label=f"Incoming/Outgoing {env}", 
        alpha=0.6, 
        color=colors[env]
    )
plt.xlabel("Attenuation")
plt.ylabel("Signal Strength (dBm)")
plt.title("Signal Strength vs. Attenuation (by Incoming/Outgoing)")
plt.legend()
plt.show()
grouped = data.groupby('Call Type')
environment_subsets = {env: group for env, group in grouped}
for env, subset in environment_subsets.items():
    mean_strength = subset['Signal Strength (dBm)'].mean()
    print(f"Mean Signal Strength (Call Type {env}): {mean_strength}")
colors = ['#FF9999', '#66B2FF']  
for env, subset in environment_subsets.items():
    plt.scatter(
        subset['Attenuation'], 
        subset['Signal Strength (dBm)'], 
        label=f"Call Type {env}", 
        alpha=0.6, 
        color=colors[env]
    )
plt.xlabel("Attenuation")
plt.ylabel("Signal Strength (dBm)")
plt.title("Signal Strength vs. Attenuation (by Call Type)")
plt.legend()
plt.show()
grouped = data.groupby('Environment')
environment_subsets = {env: group for env, group in grouped}
for env, subset in environment_subsets.items():
    mean_strength = subset['SNR'].mean()
    print(f"Mean SNR (Environment {env}): {mean_strength}")
colors = ['#FF9999', '#66B2FF', '#99FF99', '#CC99FF']  
for env, subset in environment_subsets.items():
    plt.scatter(
        subset['SNR'], 
        subset['Signal Strength (dBm)'], 
        label=f"Environment {env}", 
        alpha=0.6, 
        color=colors[env]
    )

# Customize the plot
plt.xlabel("SNR")
plt.ylabel("Signal Strength (dBm)")
plt.title("Signal Strength vs.SNR (by Environment)")
plt.legend()
plt.show()






