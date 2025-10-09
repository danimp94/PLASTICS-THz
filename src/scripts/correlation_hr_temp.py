import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Threshold for correlation
correlation_threshold = 0.7

# Paths
base_path = os.path.join(os.path.dirname(__file__), '..', '..')
characteristics_path = os.path.join(base_path, 'data', 'characteristics.csv')
processed_path = os.path.join(base_path, 'data', 'experiment_5_plastics', 'processed')
output_path = os.path.join(os.path.dirname(__file__), 'correlation_results')

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load thickness data
characteristics = pd.read_csv(characteristics_path, delimiter=';')
thickness_map = characteristics.groupby('sample')['thickness'].mean().to_dict()

# Temperature and Humidity data from README (test_id to (temp, hr))
test_temp_hr = {
    1: (19.7, 33.5), 2: (20.0, 34.1), 3: (20.7, 33.0), 4: (21.5, 34.2), 5: (22.0, 34.0),
    6: (22.4, 33.5), 7: (22.6, 32.8), 8: (22.6, 32.6), 9: (22.6, 31.8), 10: (22.4, 31.3),
    11: (22.4, 31.4), 12: (22.4, 31.0), 13: (20.0, 32.2), 14: (21.0, 33.3), 15: (21.5, 33.6),
    16: (22.1, 32.7), 17: (22.7, 31.8), 18: (22.8, 31.4), 19: (22.7, 31.0), 20: (22.8, 31.0),
    21: (22.7, 31.0), 22: (22.6, 31.0), 23: (22.6, 31.3), 24: (22.2, 32.1), 25: (19.7, 35.1),
    26: (20.8, 34.7), 27: (21.7, 34.3), 28: (22.4, 34.2), 29: (22.9, 34.1), 30: (22.9, 33.3),
    31: (23.3, 33.0), 32: (23.4, 32.3), 33: (23.4, 32.5), 34: (23.7, 32.5), 35: (23.9, 32.2),
    36: (23.6, 32.1), 37: (20.7, 40.6), 38: (21.0, 40.7), 39: (22.0, 41.8), 40: (22.4, 41.5),
    41: (22.7, 42.6), 42: (22.9, 41.8), 43: (22.5, 41.8), 44: (22.9, 41.3), 45: (23.5, 40.3),
    46: (23.5, 39.8), 47: (23.6, 40.0), 48: (23.2, 39.2), 49: (21.8, 32.1), 50: (22.4, 31.6),
    51: (23.1, 31.7), 52: (23.5, 30.6), 53: (24.0, 30.0), 54: (23.9, 28.5), 55: (24.2, 29.0),
    56: (24.3, 28.5), 57: (24.2, 28.0), 58: (24.2, 27.8), 59: (23.2, 25.4), 60: (22.8, 26.6)
}

# Mapping of Sample to test_id based on README (to handle order changes)
sample_to_test = {
    'A1_1': 1, 'B1_2': 2, 'C1_3': 3, 'D1_4': 4, 'E1_5': 5, 'F1_6': 6, 'G1_7': 7, 'H1_8': 8, 'I1_9': 9, 'J1_10': 10, 'L1_11': 11, 'O1_12': 12,
    'A2_13': 13, 'B2_14': 14, 'C2_15': 15, 'D2_16': 16, 'E2_17': 17, 'F2_18': 18, 'G2_19': 19, 'H2_20': 20, 'I2_21': 21, 'J2_22': 22, 'L2_23': 23, 'O2_24': 24,
    'A3_25': 25, 'B3_26': 26, 'C3_27': 27, 'D3_28': 28, 'E3_29': 29, 'F3_30': 30, 'G3_31': 31, 'H3_32': 32, 'I3_33': 33, 'J3_34': 34, 'L3_35': 35, 'O3_36': 36,
    'G4_37': 37, 'H4_38': 38, 'I4_39': 39, 'J4_40': 40, 'L4_41': 41, 'O4_42': 42, 'A4_43': 43, 'B4_44': 44, 'C4_45': 45, 'D4_46': 46, 'E4_47': 47, 'F4_48': 48,
    'G5_49': 49, 'H5_50': 50, 'I5_51': 51, 'J5_52': 52, 'L5_53': 53, 'O5_54': 54, 'A5_55': 55, 'B5_56': 56, 'C5_57': 57, 'D5_58': 58, 'E5_59': 59, 'F5_60': 60
}

# Load all processed CSV files
data_frames = []
for file in os.listdir(processed_path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(processed_path, file), delimiter=';')
        data_frames.append(df)

df = pd.concat(data_frames, ignore_index=True)

# Add thickness column based on sample type (first character)
df['Sample Type'] = df['Sample'].str[0]
df['Thickness (mm)'] = df['Sample Type'].map(thickness_map)

# Map test_id using the sample_to_test dictionary
df['Test ID'] = df['Sample'].map(sample_to_test)

# Map temp and hr to each row
df['Temp (°C)'] = df['Test ID'].map(lambda x: test_temp_hr.get(x, (None, None))[0])
df['HR (%)'] = df['Test ID'].map(lambda x: test_temp_hr.get(x, (None, None))[1])

# Calculate average temp and hr per sample type
temp_hr_map = df.groupby('Sample Type')[['Temp (°C)', 'HR (%)']].mean().to_dict('index')
df['AvgTemp (°C)'] = df['Sample Type'].map(lambda x: temp_hr_map[x]['Temp (°C)'])
df['AvgHR (%)'] = df['Sample Type'].map(lambda x: temp_hr_map[x]['HR (%)'])

# Remove rows with missing values
df_clean = df.dropna(subset=['AvgTemp (°C)', 'AvgHR (%)', 'HG (mV)', 'LG (mV)', 'Frequency (GHz)'])

# Get unique frequencies
frequencies = sorted(df_clean['Frequency (GHz)'].unique())

print(f"Total samples: {len(df_clean)}")
print(f"Unique frequencies: {len(frequencies)}")
print(f"Frequency range: {frequencies[0]} - {frequencies[-1]} GHz")
print(f"\nSaving correlation matrices to: {output_path}")

# List to store frequencies with low correlation (e.g., between AvgTemp and HG)
low_corr_frequencies = []

# Create correlation matrices for each frequency
for freq in frequencies:
    freq_df = df_clean[df_clean['Frequency (GHz)'] == freq]
    
    # Calculate correlation matrix for AvgTemp, AvgHR, HG, LG
    corr_matrix = freq_df[['AvgTemp (°C)', 'AvgHR (%)', 'HG (mV)', 'LG (mV)']].corr()
    
    # Get correlation between AvgTemp and HG (example)
    corr_temp_hg = corr_matrix.loc['AvgTemp (°C)', 'HG (mV)']

    # Check if absolute correlation is less than threshold
    if abs(corr_temp_hg) < correlation_threshold:
        low_corr_frequencies.append((freq, corr_temp_hg))
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                fmt='.3f', square=True, linewidths=0.5)
    plt.title(f'Correlation Matrix: Temp/HR vs HG/LG\nFrequency: {freq} GHz', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_path, f'correlation_hr_temp_freq_{freq}_GHz.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: correlation_hr_temp_freq_{freq}_GHz.png")

# 1. Total correlation matrix for AvgTemp, AvgHR, HG, LG (all frequencies)
corr_matrix_4var = df_clean[['AvgTemp (°C)', 'AvgHR (%)', 'HG (mV)', 'LG (mV)']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_4var, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            fmt='.3f', square=True, linewidths=0.5)
plt.title('Total Correlation Matrix: Temp/HR vs HG/LG\n(All Frequencies)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
output_file_4var = os.path.join(output_path, 'correlation_total_temp_hr_hg_lg.png')
plt.savefig(output_file_4var, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: correlation_total_temp_hr_hg_lg.png")

# 2. Total correlation matrix including Frequency
corr_matrix_5var = df_clean[['AvgTemp (°C)', 'AvgHR (%)', 'HG (mV)', 'LG (mV)', 'Frequency (GHz)']].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix_5var, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            fmt='.3f', square=True, linewidths=0.5)
plt.title('Total Correlation Matrix: All Variables\n(Temp, HR, HG, LG, Frequency)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
output_file_5var = os.path.join(output_path, 'correlation_hr_temp_total_all_variables.png')
plt.savefig(output_file_5var, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: correlation_hr_temp_total_all_variables.png")

# Print list of frequencies with correlation < threshold
print(f"\nFrequencies with |correlation(AvgTemp, HG)| < {correlation_threshold}:")
if low_corr_frequencies:
    for freq, corr in low_corr_frequencies:
        print(f"- {freq} GHz: {corr:.3f}")
else:
    print("None found.")

print("\nList of all frequencies with low correlation: ")
# Print as plain floats, not np.float64
print([float(freq) for freq, _ in low_corr_frequencies])

print(f"\nCorrelation analysis completed! {len(frequencies)} individual + 2 total files saved.")




# Calculate absolute correlations between AvgHR (%) and HG (mV), and AvgTemp (°C) and HG (mV) for each frequency (Pearson and Spearman)
hr_hg_corrs_pearson = []
temp_hg_corrs_pearson = []
hr_hg_corrs_spearman = []
temp_hg_corrs_spearman = []

for freq in frequencies:
    freq_df = df_clean[df_clean['Frequency (GHz)'] == freq]
    # Pearson
    corr_matrix_pearson = freq_df[['AvgTemp (°C)', 'AvgHR (%)', 'HG (mV)', 'LG (mV)']].corr(method='pearson')
    hr_hg_corr_pearson = corr_matrix_pearson.loc['AvgHR (%)', 'HG (mV)']
    temp_hg_corr_pearson = corr_matrix_pearson.loc['AvgTemp (°C)', 'HG (mV)']
    hr_hg_corrs_pearson.append((freq, hr_hg_corr_pearson))
    temp_hg_corrs_pearson.append((freq, temp_hg_corr_pearson))
    # Spearman
    corr_matrix_spearman = freq_df[['AvgTemp (°C)', 'AvgHR (%)', 'HG (mV)', 'LG (mV)']].corr(method='spearman')
    hr_hg_corr_spearman = corr_matrix_spearman.loc['AvgHR (%)', 'HG (mV)']
    temp_hg_corr_spearman = corr_matrix_spearman.loc['AvgTemp (°C)', 'HG (mV)']
    hr_hg_corrs_spearman.append((freq, hr_hg_corr_spearman))
    temp_hg_corrs_spearman.append((freq, temp_hg_corr_spearman))

# Pearson stats
abs_hr_hg_corrs_pearson = [(freq, abs(corr)) for freq, corr in hr_hg_corrs_pearson]
max_hr_hg_pearson = max(abs_hr_hg_corrs_pearson, key=lambda x: x[1])
avg_hr_hg_pearson = sum(corr for _, corr in abs_hr_hg_corrs_pearson) / len(abs_hr_hg_corrs_pearson)
abs_temp_hg_corrs_pearson = [(freq, abs(corr)) for freq, corr in temp_hg_corrs_pearson]
max_temp_hg_pearson = max(abs_temp_hg_corrs_pearson, key=lambda x: x[1])
avg_temp_hg_pearson = sum(corr for _, corr in abs_temp_hg_corrs_pearson) / len(abs_temp_hg_corrs_pearson)

# Spearman stats
abs_hr_hg_corrs_spearman = [(freq, abs(corr)) for freq, corr in hr_hg_corrs_spearman]
max_hr_hg_spearman = max(abs_hr_hg_corrs_spearman, key=lambda x: x[1])
avg_hr_hg_spearman = sum(corr for _, corr in abs_hr_hg_corrs_spearman) / len(abs_hr_hg_corrs_spearman)
abs_temp_hg_corrs_spearman = [(freq, abs(corr)) for freq, corr in temp_hg_corrs_spearman]
max_temp_hg_spearman = max(abs_temp_hg_corrs_spearman, key=lambda x: x[1])
avg_temp_hg_spearman = sum(corr for _, corr in abs_temp_hg_corrs_spearman) / len(abs_temp_hg_corrs_spearman)

print("\n--- Correlation Summary (Pearson) ---")
print(f"Max |correlation(HR, Transmission)|: {max_hr_hg_pearson[1]:.3f} at {max_hr_hg_pearson[0]} GHz")
print(f"Avg |correlation(HR, Transmission)|: {avg_hr_hg_pearson:.3f}")
print(f"Max |correlation(Temp, Transmission)|: {max_temp_hg_pearson[1]:.3f} at {max_temp_hg_pearson[0]} GHz")
print(f"Avg |correlation(Temp, Transmission)|: {avg_temp_hg_pearson:.3f}")

print("\n--- Correlation Summary (Spearman) ---")
print(f"Max |correlation(HR, Transmission)|: {max_hr_hg_spearman[1]:.3f} at {max_hr_hg_spearman[0]} GHz")
print(f"Avg |correlation(HR, Transmission)|: {avg_hr_hg_spearman:.3f}")
print(f"Max |correlation(Temp, Transmission)|: {max_temp_hg_spearman[1]:.3f} at {max_temp_hg_spearman[0]} GHz")
print(f"Avg |correlation(Temp, Transmission)|: {avg_temp_hg_spearman:.3f}")
