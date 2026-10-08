# README

## Overview

This repository contains scripts and Jupyter notebooks for preprocessing, training machine learning models, and visualizing spectroscopy data. The tools are designed to handle LVM and CSV files, perform data processing, clustering, model training, and generate various plots for data visualization and analysis.

## Directory Structure

```
src/
├── scripts/
│   ├── train_v2.py
│   ├── preprocessing.py
│   ├── plotter.py
│   ├── correlation_hr_temp.py
│   └── correlation_thickness.py
├── nb/
│   ├── train.ipynb
│   ├── train_v2.ipynb
│   ├── stabilization.ipynb
│   └── visualization_playground.ipynb
└── README.md
```
## Quick Start

### Installation
```sh
git clone https://github.com/danimp94/PLASTICS-THz.git
cd PLASTICS-THz
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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

### 3. Training (Experiment 5)

**File:** `train_v2.py`

Leave-one-day-out training over the 5 experiment-5 days: shared nested frequency selection (RF+GB+LR), 5 models (RF, NB, LR, GB, SVM) on baseline and alpha (Beer–Lambert) arms. Writes per-fold/summary/stability CSVs, accuracy/stability/selection charts and pooled confusion matrices to `results/exp_5_v2/`.

#### Usage

```sh
python src/scripts/train_v2.py
```

Set `SMOKE = True` in the script for a 1-fold, K=[10,3] wiring check.

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

### 4. LODO Training (Experiment 5)

**File:** `nb/train_v2.ipynb`

Notebook entry point for the same pipeline as `train_v2.py`: it imports the canonical implementation from `src/scripts/train_v2.py` and runs it (`SMOKE = False` for the full run). Run top to bottom; the import cell asserts the right module is loaded. Extra cells below the driver are viz-only exploration (PCA, 3D frequency plots).

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
# - train_v2.ipynb for the Experiment-5 LODO pipeline
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
- pyarrow

## Output Structure

The notebooks and scripts generate organized output in the following structure:

```
results/
├── exp_5_v2/              # LODO training outputs: lodo_per_fold/summary/stability.csv,
│                          # accuracy/stability/selection charts, pooled confusion matrices
├── pca_models/            # PCA visualization plots
├── conf_matrix/           # Confusion matrices
└── freq_viz/              # Frequency-specific visualizations (created by train_v2.ipynb)
```

## Contact

For any questions or issues, please contact Daniel Moreno at danmoren@pa.uc3m.es