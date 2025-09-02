# README

## Overview

This repository contains scripts and Jupyter notebooks for preprocessing, training machine learning models, and visualizing spectroscopy data. The tools are designed to handle LVM and CSV files, perform data processing, clustering, model training, and generate various plots for data visualization and analysis.

## Directory Structure

```
src/
├── scripts/
│   ├── preprocessing.py
│   └── plotter.py
├── nb/
│   ├── train.ipynb
│   ├── stabilization.ipynb
│   └── visualization_playground.ipynb
└── README.md
```

## Scripts

### 1. Preprocessing

**File:** `preprocessing.py`

This script processes LVM files, converts them to CSV, and performs various data manipulations such as discarding data, merging files, and calculating averages.

#### Usage

```sh
python preprocessing.py <command> [options]
```

#### Commands

- `process_single <input_file> <output_dir>`: Process a single LVM file.
- `process_multiple <input_dir> <output_dir>`: Process multiple LVM files in a directory.
- `concatenate <input_dir> <output_dir>`: Concatenate CSV files.
- `remove_columns <input_dir> <output_dir> <num_columns> <position>`: Remove columns from CSV files.
- `add_thickness <input_dir> <output_dir> <characteristics_file>`: Add thickness column to CSV files.
- `merge <input_dir> <output_file>`: Merge all CSV files in a directory.
- `calculate_averages <input_file> <output_file>`: Calculate averages and standard deviations.
- `calculate_transmittance <input_file> <output_dir>`: Calculate transmittance.
- `calculate_averages_and_dispersion <input_file> <output_file> [--data_percentage <percentage>]`: Calculate averages and dispersion.

### 2. Plotting

**File:** `plotter.py`

This script generates various plots such as heatmaps, overlays, and transmittance plots from the processed data.

#### Usage

```sh
python plotter.py <command> [options]
```

#### Commands

- `heatmap <data_files> <channel_indices> [--plot_together]`: Plot heatmap of spectroscopy data.
- `overlay <data_files> <channel_indices>`: Plot overlay of spectroscopy data.
- `overlay_avg <data_files> <channel_indices>`: Plot overlay of averaged spectroscopy data.
- `plot_transmittance <file_path> [samples]`: Plot transmittance from CSV file.

## Jupyter Notebooks

### 1. Training Notebook

**File:** `nb/train.ipynb`

This comprehensive notebook handles the complete machine learning pipeline for spectroscopy data classification:

#### Features
- **Data Loading & Preprocessing**: Load data from directories, handle time windows, and pivot frequency values to columns
- **Feature Engineering**: Add differential features, apply scaling, and dimensionality reduction techniques
- **Model Training**: Train multiple classifiers including:
  - Random Forest
  - Naive Bayes
  - Logistic Regression
  - Gradient Boosting
  - Support Vector Machine
- **Model Evaluation**: Performance metrics, confusion matrices, and feature importance analysis
- **Visualization**: PCA plots (1D, 2D, 3D), frequency-specific 3D plots
- **Results Export**: Save model results, confusion matrices, and visualizations

#### Key Functions
- Data preprocessing with configurable options
- Multiple model training and comparison
- Feature importance extraction
- PCA and dimensionality reduction analysis
- Performance evaluation with AIC/BIC criteria

### 2. Stabilization Analysis

**File:** `nb/stabilization.ipynb`

This notebook focuses on analyzing signal stabilization in spectroscopy measurements:

#### Features
- Signal stability analysis over time
- Stabilization time calculation
- Time-series visualization
- Signal quality assessment

### 3. Visualization Playground

**File:** `nb/visualization_playground.ipynb`

An experimental notebook for developing and testing various visualization techniques:

#### Features
- Interactive plotting experiments
- Custom visualization development
- Data exploration tools
- Plot customization testing

## Example Commands

### Preprocessing

```sh
python preprocessing.py process_single ../../data/experiment_1_plastics/raw/sample1.lvm ../../data/experiment_1_plastics/processed/
python preprocessing.py process_multiple ../../data/experiment_1_plastics/raw/ ../../data/experiment_1_plastics/processed/
python preprocessing.py concatenate ../../data/experiment_1_plastics/processed_full/dispersion_2/ ../../data/experiment_1_plastics/processed_full/dispersion_2/conc/
python preprocessing.py remove_columns ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/processed/ 4 last
python preprocessing.py add_thickness ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/characteristics.csv
python preprocessing.py merge ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/processed/merged.csv
python preprocessing.py calculate_averages ../../data/experiment_1_plastics/processed/merged.csv ../../data/experiment_1_plastics/processed/averages.csv
python preprocessing.py calculate_transmittance ../../data/experiment_1_plastics/processed/averages.csv ../../data/experiment_1_plastics/processed/
python preprocessing.py calculate_averages_and_dispersion ../../data/experiment_1_plastics/processed/averages.csv ../../data/experiment_1_plastics/processed/averages_dispersion.csv --data_percentage 50
```

### Plotting

```sh
python plotter.py heatmap ../../data/experiment_1_plastics/processed/*.csv 2 --plot_together
python plotter.py overlay ../../data/experiment_1_plastics/processed/*.csv 2
python plotter.py overlay_avg ../../data/experiment_1_plastics/processed/*.csv 2
python plotter.py plot_transmittance ../../data/experiment_1_plastics/processed/result/transmittance_results.csv A1 B1 C1
```

### Notebook Usage

To run the Jupyter notebooks:

```sh
# Navigate to the notebook directory
cd nb/

# Start Jupyter Lab or Notebook
jupyter lab
# or
jupyter notebook

# Open the desired notebook:
# - train.ipynb for machine learning pipeline
# - stabilization.ipynb for signal analysis
# - visualization_playground.ipynb for experimental plots
```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- mplcursors
- jupyter
- joblib
- scipy

## Installation

Install the required Python packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn mplcursors jupyter joblib scipy
```

## Output Structure

The notebooks and scripts generate organized output in the following structure:

```
results/
├── pca_models/          # PCA visualization plots
├── conf_matrix/         # Confusion matrices
├── feature_importance_detailed/  # Feature importance plots
├── freq_viz/           # Frequency-specific visualizations
└── exp_5/              # Experiment results and metrics
```

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please contact Daniel Moreno at danmoren@pa.uc3m.es