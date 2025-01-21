import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import pickle

def plot3D():
    try:
        # Load data dictionary
        with open('notebook/pca_models/pca_viz_250_300_320_330_340_350_360_370_380_390_400_410_420_430_440_450_460_470_480_490_500_510_520_530_540.pkl', 'rb') as file:
            data_dict = pickle.load(file)
            
        # Extract components
        X = data_dict['X_transformed']
        labels = data_dict['labels']
        var_ratio = data_dict['explained_variance_ratio']
        feature_names = data_dict['feature_names']
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot with label colors
        unique_labels = labels.unique()
        n_labels = len(unique_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, n_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                      c=[color], label=label, alpha=0.8)
        
        # Add labels with explained variance
        ax.set_xlabel(f'{feature_names[0]}\n({var_ratio[0]:.1%} variance)')
        ax.set_ylabel(f'{feature_names[1]}\n({var_ratio[1]:.1%} variance)')
        ax.set_zlabel(f'{feature_names[2]}\n({var_ratio[2]:.1%} variance)')
        
        plt.title('PCA Components Visualization', pad=20)
        plt.legend(bbox_to_anchor=(1.15, 1), title='Classes')
        
        ax.grid(True)
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}\nType: {type(e)}")

if __name__ == "__main__":
    plot3D()