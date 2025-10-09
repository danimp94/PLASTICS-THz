import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Threshold for correlation
correlation_threshold = 0.6

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

# Remove rows with missing values
df_clean = df.dropna(subset=['Thickness (mm)', 'HG (mV)', 'LG (mV)', 'Frequency (GHz)'])

# Get unique frequencies
frequencies = sorted(df_clean['Frequency (GHz)'].unique())

print(f"Total samples: {len(df_clean)}")
print(f"Unique frequencies: {len(frequencies)}")
print(f"Frequency range: {frequencies[0]} - {frequencies[-1]} GHz")
print(f"\nSaving correlation matrices to: {output_path}")

# List to store frequencies with low and high correlation
low_corr_frequencies = []
high_corr_frequencies = []

# Create correlation matrices for each frequency
for freq in frequencies:
    freq_df = df_clean[df_clean['Frequency (GHz)'] == freq]
    
    # Calculate correlation matrix
    corr_matrix = freq_df[['Thickness (mm)', 'HG (mV)', 'LG (mV)']].corr()
    
    # Get correlation between Thickness and HG
    corr_thickness_hg = corr_matrix.loc['Thickness (mm)', 'HG (mV)']

    # Check if absolute correlation is less than threshold
    if abs(corr_thickness_hg) < correlation_threshold:
        low_corr_frequencies.append((freq, corr_thickness_hg))
    else:
        high_corr_frequencies.append((freq, corr_thickness_hg))
    
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                fmt='.3f', square=True, linewidths=0.5)
    plt.title(f'Correlation Matrix: Thickness vs HG/LG\nFrequency: {freq} GHz', 
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_path, f'correlation_freq_{freq}_GHz.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: correlation_freq_{freq}_GHz.png")

# 1. Total correlation matrix for Thickness, HG, LG (all frequencies)
corr_matrix_3var = df_clean[['Thickness (mm)', 'HG (mV)', 'LG (mV)']].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix_3var, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            fmt='.3f', square=True, linewidths=0.5)
plt.title('Total Correlation Matrix: Thickness vs HG/LG\n(All Frequencies)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
output_file_3var = os.path.join(output_path, 'correlation_total_thickness_hg_lg.png')
plt.savefig(output_file_3var, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: correlation_total_thickness_hg_lg.png")

# 2. Total correlation matrix including Frequency
corr_matrix_4var = df_clean[['Thickness (mm)', 'HG (mV)', 'LG (mV)', 'Frequency (GHz)']].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix_4var, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            fmt='.3f', square=True, linewidths=0.5)
plt.title('Total Correlation Matrix: All Variables\n(Thickness, HG, LG, Frequency)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
output_file_4var = os.path.join(output_path, 'correlation_total_all_variables.png')
plt.savefig(output_file_4var, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: correlation_total_all_variables.png")

# Print list of frequencies with correlation < threshold
print(f"\nFrequencies with low correlation(Thickness, HG)| < {correlation_threshold}:")
if low_corr_frequencies:
    for freq, corr in low_corr_frequencies:
        print(f"- {freq} GHz: {corr:.3f}")
else:
    print("None found.")

#Print list of frequencies with correlation >= threshold
print(f"\nFrequencies with high correlation(Thickness, HG)| >= {correlation_threshold}:")
if high_corr_frequencies:
    for freq, corr in high_corr_frequencies:
        print(f"- {freq} GHz: {corr:.3f}")
else:
    print("None found.")

print("\nList of all frequencies with low correlation: ")
# Print as plain floats, not np.float64
print([float(freq) for freq, _ in low_corr_frequencies])

print(f"\nCorrelation analysis completed! {len(frequencies)} individual + 2 total files saved.")

# Calculate absolute correlations between Thickness (mm) and HG (mV) for each frequency (Pearson and Spearman)
thickness_hg_corrs_pearson = []
thickness_hg_corrs_spearman = []

for freq in frequencies:
    freq_df = df_clean[df_clean['Frequency (GHz)'] == freq]
    # Pearson
    corr_matrix_pearson = freq_df[['Thickness (mm)', 'HG (mV)', 'LG (mV)']].corr(method='pearson')
    thickness_hg_corr_pearson = corr_matrix_pearson.loc['Thickness (mm)', 'HG (mV)']
    thickness_hg_corrs_pearson.append((freq, thickness_hg_corr_pearson))
    # Spearman
    corr_matrix_spearman = freq_df[['Thickness (mm)', 'HG (mV)', 'LG (mV)']].corr(method='spearman')
    thickness_hg_corr_spearman = corr_matrix_spearman.loc['Thickness (mm)', 'HG (mV)']
    thickness_hg_corrs_spearman.append((freq, thickness_hg_corr_spearman))

# Pearson stats
abs_thickness_hg_corrs_pearson = [(freq, abs(corr)) for freq, corr in thickness_hg_corrs_pearson]
max_thickness_hg_pearson = max(abs_thickness_hg_corrs_pearson, key=lambda x: x[1])
avg_thickness_hg_pearson = sum(corr for _, corr in abs_thickness_hg_corrs_pearson) / len(abs_thickness_hg_corrs_pearson)

# Spearman stats
abs_thickness_hg_corrs_spearman = [(freq, abs(corr)) for freq, corr in thickness_hg_corrs_spearman]
max_thickness_hg_spearman = max(abs_thickness_hg_corrs_spearman, key=lambda x: x[1])
avg_thickness_hg_spearman = sum(corr for _, corr in abs_thickness_hg_corrs_spearman) / len(abs_thickness_hg_corrs_spearman)

print("\n--- Correlation Summary (Pearson) ---")
print(f"Max |correlation(Thickness, Transmission)|: {max_thickness_hg_pearson[1]:.3f} at {max_thickness_hg_pearson[0]} GHz")
print(f"Avg |correlation(Thickness, Transmission)|: {avg_thickness_hg_pearson:.3f}")

print("\n--- Correlation Summary (Spearman) ---")
print(f"Max |correlation(Thickness, Transmission)|: {max_thickness_hg_spearman[1]:.3f} at {max_thickness_hg_spearman[0]} GHz")
print(f"Avg |correlation(Thickness, Transmission)|: {avg_thickness_hg_spearman:.3f}")
