import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def read_lvm(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                line = line.replace(',', '.').strip()
                line_data = list(map(float, line.split('\t')))
                if len(line_data) == 7:
                    data.append(line_data)
                else:
                    print(f"Skipping line with unexpected column count: {line_data}")
            except ValueError as e:
                print(f"Skipping line due to ValueError: {e}")
                continue
    return np.array(data) if data else np.array([])

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path, delimiter=';')
        data = df.to_numpy()
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([])

def plot_data_heatmap(data_files, channel_indices, plot_together=True):
    if plot_together:
        plt.figure(figsize=(16, 12))

    for file_path in data_files:
        if not plot_together:
            plt.figure(figsize=(16, 12))

        data = read_csv(file_path)

        if data.size > 0:
            x_data = data[:, 0]
            for index in channel_indices:
                if index < data.shape[1]:
                    channel_data = data[:, index]
                    mask = (x_data != 600)
                    x_data_filtered = x_data[mask]
                    channel_data_filtered = channel_data[mask]

                    label = f'Channel {index} from {os.path.basename(file_path)}'
                    heatmap, xedges, yedges = np.histogram2d(x_data_filtered, channel_data_filtered, bins=50)
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

                else:
                    print(f"Invalid channel index: {index} for file {file_path}")

        if not plot_together:
            plt.title(f'Heatmap of LVM Data (Selected Channels vs Frequency) - {os.path.basename(file_path)}', fontsize=20)
            plt.xlabel('Frequency (GHz)', fontsize=16)
            plt.ylabel('Values (mV)', fontsize=16)
            plt.grid(True)

            save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(save_dir, f'heatmap_plot_{base_name}.png')
            plt.savefig(save_path, dpi=300)
            print(f"Heatmap plot saved as: {save_path}")

    if plot_together:
        plt.title('Heatmap of LVM Data (Selected Channels vs Frequency)', fontsize=20)
        plt.xlabel('Frequency (GHz)', fontsize=16)
        plt.ylabel('Values (mV)', fontsize=16)
        plt.grid(True)

        save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        base_name = os.path.splitext(os.path.basename(data_files[0]))[0]
        save_path = os.path.join(save_dir, f'heatmap_plot_{base_name}_combined.png')
        plt.savefig(save_path, dpi=300)
        print(f"Combined heatmap plot saved as: {save_path}")

def plot_data_overlay(data_files, channel_indices):
    plt.figure(figsize=(16, 12))

    for file_path in data_files:
        data = read_csv(file_path)

        if data.size > 0:
            x_data = data[:, 0]

            for index in channel_indices:
                if index < data.shape[1]:
                    channel_data = data[:, index]
                    mask = (x_data != 600) & (x_data != 210)
                    x_filtered = x_data[mask]
                    y_filtered = channel_data[mask]

                    label = f'Channel {index} from {os.path.basename(file_path)}'
                    plt.scatter(x_filtered, y_filtered, label=label, alpha=0.6)

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Values (mV)')
    plt.title('Spectroscopy Sample Overlay (Scattered)')
    plt.legend()
    plt.show()

def plot_data_overlay_average(data_files, channel_indices):
    plt.figure(figsize=(16, 12))

    for file_path in data_files:
        data = read_csv(file_path)

        if data.size > 0:
            x_data = data[:, 0]
            aggregated_data = {}

            for index in channel_indices:
                if index < data.shape[1]:
                    channel_data = data[:, index]
                    mask = np.ones_like(x_data, dtype=bool)
                    x_filtered = x_data[mask]
                    y_filtered = channel_data[mask]

                    for x, y in zip(x_filtered, y_filtered):
                        if x not in aggregated_data:
                            aggregated_data[x] = []
                        aggregated_data[x].append(y)

            avg_data = {x: np.mean(y) for x, y in aggregated_data.items()}
            sorted_freqs = sorted(avg_data.keys())
            sorted_avg_values = [avg_data[freq] for freq in sorted_freqs]

            label = f'Average from {os.path.basename(file_path)}'
            plt.plot(sorted_freqs, sorted_avg_values, label=label)

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Values (mV)')
    plt.title('Spectroscopy Sample Overlay (Averaged)')
    plt.legend()

    save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in data_files]
    base_name = "_".join(base_names)
    save_path = os.path.join(save_dir, f'overlay_average_plot_{base_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Averaged overlay plot saved as: {save_path}")

    plt.show()

def plot_transmittance(file_path, selected_samples=None):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=';')

        # If no samples are selected, plot all samples
        if selected_samples is None:
            selected_samples = df['Sample'].unique()

        # Filter the DataFrame based on the selected samples
        filtered_df = df[df['Sample'].isin(selected_samples)]

        # Plot the data
        plt.figure(figsize=(16, 12))
        for sample in selected_samples:
            sample_df = filtered_df[filtered_df['Sample'] == sample]
            plt.plot(sample_df['Frequency (GHz)'], sample_df['HG (mV) mean Transmittance'], label=f'{sample} HG Transmittance', linestyle='-', marker='o')
            plt.plot(sample_df['Frequency (GHz)'], sample_df['LG (mV) mean Transmittance'], label=f'{sample} LG Transmittance', linestyle='--', marker='x')

        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Transmittance Ratio')
        plt.title('Transmittance vs Frequency')
        plt.legend()

        # Save the plot
        save_dir = os.path.join(os.path.dirname(file_path), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f'transmittance_plot_{base_name}.png')
        plt.savefig(save_path, dpi=300)
        print(f"Transmittance plot saved as: {save_path}")

        plt.show()

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description="Plot spectroscopy data.")
    subparsers = parser.add_subparsers(dest="command")

    parser_heatmap = subparsers.add_parser("heatmap", help="Plot heatmap of spectroscopy data")
    parser_heatmap.add_argument("data_files", nargs='+', help="List of CSV files to plot")
    parser_heatmap.add_argument("channel_indices", nargs='+', type=int, help="Indices of channels to plot")
    parser_heatmap.add_argument("--plot_together", action="store_true", help="Plot all data together in one heatmap")

    parser_overlay = subparsers.add_parser("overlay", help="Plot overlay of spectroscopy data")
    parser_overlay.add_argument("data_files", nargs='+', help="List of CSV files to plot")
    parser_overlay.add_argument("channel_indices", nargs='+', type=int, help="Indices of channels to plot")

    parser_overlay_avg = subparsers.add_parser("overlay_avg", help="Plot overlay of averaged spectroscopy data")
    parser_overlay_avg.add_argument("data_files", nargs='+', help="List of CSV files to plot")
    parser_overlay_avg.add_argument("channel_indices", nargs='+', type=int, help="Indices of channels to plot")

    parser_transmittance = subparsers.add_parser("plot_transmittance", help="Plot transmittance from CSV file")
    parser_transmittance.add_argument("file_path", help="Path to the CSV file containing transmittance data")
    parser_transmittance.add_argument("samples", nargs='*', help="List of samples to plot (e.g., A1 B1 C1)")

    args = parser.parse_args()

    if args.command == "heatmap":
        plot_data_heatmap(args.data_files, args.channel_indices, args.plot_together)
    elif args.command == "overlay":
        plot_data_overlay(args.data_files, args.channel_indices)
    elif args.command == "overlay_avg":
        plot_data_overlay_average(args.data_files, args.channel_indices)
    elif args.command == "plot_transmittance":
        plot_transmittance(args.file_path, args.samples if args.samples else None)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

    # input_file_path = '../../data/experiment_1_plastics/processed/result/transmittance_results.csv'
    # plot_transmittance(input_file_path)

# # Input file path
#  ../../data/experiment_1_plastics/processed/merged_averages_std_dev.csv

# # Input directory
#  ../../data/experiment_1_plastics/processed/

# # Output file path
#  ../../data/experiment_1_plastics/processed/plots/

# # Example commands
# py plotter.py heatmap ../../data/experiment_1_plastics/processed/*.csv 2 --plot_together
# py plotter.py overlay ../../data/experiment_1_plastics/processed/*.csv 2
# py plotter.py overlay_avg ../../data/experiment_1_plastics/processed/*.csv 2
# py plotter.py plot_transmittance ../../data/experiment_1_plastics/processed/result/transmittance_results.csv A1 B1 C1