# README

## Overview

This repository contains scripts for preprocessing, clustering, and plotting spectroscopy data. The scripts are designed to handle LVM and CSV files, perform data processing, clustering, and generate various plots for data visualization.

## Directory Structure

```
src/
├── scripts/
│   ├── preprocessing.py
│   ├── cluster.py
│   ├── plotter.py
│   └── README.md
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

### 2. Clustering

**File:** `cluster.py`

This script performs clustering on the processed data using KMeans and generates 3D plots of the clusters.

#### Usage

```sh
python cluster.py
```

### 3. Plotting

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

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- mplcursors

## Installation

Install the required Python packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn mplcursors
```

## License

This project is licensed under the MIT License.

## Contact

For any questions or issues, please contact Daniel Moreno at danmoren@pa.uc3m.es
