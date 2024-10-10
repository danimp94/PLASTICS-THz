# import os
# import pandas as pd
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns

# def compute_features(df):
#     """Compute summary statistics for HG and LG columns in the DataFrame."""
#     features = {}
#     if 'HG (mV)' in df.columns and 'LG (mV)' in df.columns:
#         # Convert HG (mV) and LG (mV) columns to numeric, forcing errors to NaN
#         df['HG (mV)'] = pd.to_numeric(df['HG (mV)'], errors='coerce')
#         df['LG (mV)'] = pd.to_numeric(df['LG (mV)'], errors='coerce')
        
#         # Drop rows with NaN values in HG (mV) or LG (mV) columns
#         df = df.dropna(subset=['HG (mV)', 'LG (mV)'])
        
#         # Group by 'Sample' and compute mean and std for each group
#         grouped = df.groupby('Sample').agg(
#             HG_mean=('HG (mV)', 'mean'), HG_std=('HG (mV)', 'std'),
#             LG_mean=('LG (mV)', 'mean'), LG_std=('LG (mV)', 'std')
#         ).reset_index()
        
#         features = grouped
#     return features

# def cluster_file(file_path, output_directory, n_clusters=3):
#     if not os.path.exists(output_directory):
#         print(f"Creating output directory: {output_directory}")
#         os.makedirs(output_directory)

#     # Read the CSV file
#     df = pd.read_csv(file_path, delimiter=';')
    
#     # Print the columns of the DataFrame to debug
#     print(f"Processing file: {file_path}")
#     print(f"Columns: {df.columns.tolist()}")

#     # Compute features for the file
#     features_df = compute_features(df)
#     if features_df.empty:
#         print("No valid HG (mV) or LG (mV) data found.")
#         return

#     # Ensure the number of clusters is less than or equal to the number of samples
#     n_clusters = min(n_clusters, len(features_df))

#     # Perform clustering on the feature vectors
#     kmeans = KMeans(n_clusters=n_clusters, max_iter=1000)
#     features_df['cluster'] = kmeans.fit_predict(features_df[['HG_mean', 'HG_std', 'LG_mean', 'LG_std']])

#     # Plot the clusters
#     plt.figure(figsize=(10, 6))
#     scatter = sns.scatterplot(data=features_df, x='HG_mean', y='HG_std', hue='cluster', palette='viridis', s=100)

#     # Annotate each point with the sample name
#     for i in range(features_df.shape[0]):
#         scatter.text(features_df['HG_mean'][i], features_df['HG_std'][i], features_df['Sample'][i], 
#                      horizontalalignment='left', size='medium', color='black', weight='semibold')

#     plt.title('Cluster of Samples Based on HG and LG')
#     plt.xlabel('Mean HG (mV)')
#     plt.ylabel('Standard Deviation HG (mV)')
#     plt.legend(title='Cluster')
#     plt.grid(True)
#     plt.show()

# def main():
#     file_path = os.path.abspath('../../data/experiment_1_plastics/processed/merged_averages_std_dev.csv')
#     output_directory = os.path.abspath('../../data/experiment_1_plastics/processed/clustered/')
    
#     print(f"File path: {file_path}")
#     print(f"Output directory: {output_directory}")
    
#     if not os.path.exists(output_directory):
#         print(f"Creating output directory: {output_directory}")
#         os.makedirs(output_directory)
        
#     cluster_file(file_path, output_directory)
#     print("Clustering complete!")

# if __name__ == "__main__":
#     main()


import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def compute_features(df):
    """Compute summary statistics for HG and LG columns in the DataFrame."""
    if 'HG (mV) mean' in df.columns and 'LG (mV) mean' in df.columns and 'Thickness (mm)' in df.columns:
        # Group by 'Sample' and compute mean and std for each group
        grouped = df.groupby('Sample').agg(
            HG_mean=('HG (mV) mean', 'mean'), HG_std=('HG (mV) std', 'mean'),
            LG_mean=('LG (mV) mean', 'mean'), LG_std=('LG (mV) std', 'mean'),
            Thickness=('Thickness (mm)', 'mean')
        ).reset_index()
        
        return grouped
    else:
        return pd.DataFrame()  # Return an empty DataFrame if required columns are not present

def cluster_file(file_path, output_directory, n_clusters=3):
    if not os.path.exists(output_directory):
        print(f"Creating output directory: {output_directory}")
        os.makedirs(output_directory)

    # Read the CSV file
    df = pd.read_csv(file_path, delimiter=';')
    
    # Print the columns of the DataFrame to debug
    print(f"Processing file: {file_path}")
    print(f"Columns: {df.columns.tolist()}")

    # Compute features for the file
    features_df = compute_features(df)
    if features_df.empty:
        print("No valid HG (mV) or LG (mV) data found.")
        return

    # Ensure the number of clusters is less than or equal to the number of samples
    n_clusters = min(n_clusters, len(features_df))

    # Perform clustering on the feature vectors
    kmeans = KMeans(n_clusters=n_clusters, max_iter=10)
    features_df['cluster'] = kmeans.fit_predict(features_df[['HG_mean', 'HG_std', 'LG_mean', 'LG_std', 'Thickness']])

    # Plot the clusters in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(features_df['HG_mean'], features_df['HG_std'], features_df['Thickness'], c=features_df['cluster'], cmap='viridis', s=100)

    # Annotate each point with the sample name
    for i in range(features_df.shape[0]):
        ax.text(features_df['HG_mean'][i], features_df['HG_std'][i], features_df['Thickness'][i], features_df['Sample'][i], 
                horizontalalignment='left', size='medium', color='black', weight='semibold')

    ax.set_title('Cluster of Samples Based on HG, LG, and Thickness')
    ax.set_xlabel('Mean HG (mV)')
    ax.set_ylabel('Standard Deviation HG (mV)')
    ax.set_zlabel('Thickness (mm)')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.show()

def main():
    file_path = os.path.abspath('../../data/experiment_1_plastics/processed/result/merged_averages_std_dev.csv')
    output_directory = os.path.abspath('../../data/experiment_1_plastics/processed/clustered/')
    
    print(f"File path: {file_path}")
    print(f"Output directory: {output_directory}")
    
    if not os.path.exists(output_directory):
        print(f"Creating output directory: {output_directory}")
        os.makedirs(output_directory)
        
    cluster_file(file_path, output_directory)
    print("Clustering complete!")

if __name__ == "__main__":
    main()