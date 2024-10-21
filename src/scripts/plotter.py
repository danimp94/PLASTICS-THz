import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mplcursors
from matplotlib.widgets import Button

def get_files(file_or_directory):
    if os.path.isdir(file_or_directory):
        # List all files in the directory and filter for CSV and LVM files
        files = os.listdir(file_or_directory)
        data_files = [os.path.join(file_or_directory, f) for f in files if f.endswith('.csv') or f.endswith('.lvm')]
        return data_files
    elif os.path.isfile(file_or_directory) and (file_or_directory.endswith('.csv') or file_or_directory.endswith('.lvm')):
        return [file_or_directory]
    else:
        print(f"Invalid input: {file_or_directory}")
        return []

def read_lvm(file_path, start_line):
    data = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= start_line:  # Skip lines until reaching the data section
                try:
                    line = line.replace(',', '.').strip()  # Ensure any extra spaces are removed
                    line_data = list(map(float, line.split('\t')))
                    data.append(line_data)
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

def plot_data_heatmap(file_or_directory, channel_indices, plot_together=True):
    data_files = get_files(file_or_directory)
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

def plot_data_overlay(file_or_directory, channel_indices):
    data_files = get_files(file_or_directory)
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

def plot_data_overlay_average(file_or_directory, channel_indices):
    data_files = get_files(file_or_directory)
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

def plot_transmittance(file_or_directory, selected_samples=None):
    data_files = get_files(file_or_directory)
    
    # If no samples are selected, use all files as samples
    if selected_samples is None:
        selected_samples = [os.path.splitext(os.path.basename(file_path))[0] for file_path in data_files]

    plt.figure(figsize=(16, 12))
    lines = []

    for file_path in data_files:
        # Read the CSV file
        df = pd.read_csv(file_path, delimiter=';')

        # Get the sample name from the file name
        sample_name = os.path.splitext(os.path.basename(file_path))[0]

        # Calculate transmittance ratio using REF files as reference
        # Calculate the mean of the REF files
        ref_data = sample_name.startswith('REF')

        # Skip files that are not used as reference
        if ref_data:
            ref_files = [f for f in data_files if os.path.splitext(os.path.basename(f))[0].startswith('REF')]
            ref_df = pd.concat([pd.read_csv(f, delimiter=';') for f in ref_files], ignore_index=True)
            ref_grouped = ref_df.groupby('Frequency (GHz)')

    # Calculate transmittance for each frequency
    for file_path in data_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=';')

            # Get the sample name from the file name
            sample_name = os.path.splitext(os.path.basename(file_path))[0]

            # Skip files that are not in the selected samples
            if sample_name not in selected_samples:
                continue

            # Calculate mean for each frequency
            grouped = df.groupby('Frequency (GHz)')
            df['HG (mV) mean'] = grouped['HG (mV)'].transform('mean')
            df['LG (mV) mean'] = grouped['LG (mV)'].transform('mean')

            # Calculate Transmittance ratio
            df['HG (mV) Transmittance'] = df['HG (mV) mean'] / ref_grouped['HG (mV)'].transform('mean')
            df['LG (mV) Transmittance'] = df['LG (mV) mean'] / ref_grouped['LG (mV)'].transform('mean')       

            # Plot the data as sparse
            line1, = plt.plot(df['Frequency (GHz)'], df['HG (mV) Transmittance'], label=f'{sample_name} HG Transmittance', linestyle='-', marker='o')
            line2, = plt.plot(df['Frequency (GHz)'], df['LG (mV) Transmittance'], label=f'{sample_name} LG Transmittance', linestyle='--', marker='x')
            lines.extend([line1, line2])

        except FileNotFoundError as e:
            print(f"File not found: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Add labels, legend, and grid
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Transmittance Ratio')
    plt.title('Transmittance vs Frequency')
    legend = plt.legend()
    plt.grid(True)

    # Add interactive cursor
    cursor = mplcursors.cursor(lines, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))

    # Add button to hide/show legend
    def toggle_legend(event):
        legend.set_visible(not legend.get_visible())
        plt.draw()

    ax_button = plt.axes([0.81, 0.01, 0.1, 0.075])
    button = Button(ax_button, 'Toggle Legend')
    button.on_clicked(toggle_legend)

    # Save the plot
    save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_name = "_".join(selected_samples)
    save_path = os.path.join(save_dir, f'transmittance_plot_{base_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Transmittance plot saved as: {save_path}")

    plt.show() 

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

    input = '../../data/experiment_2_plastics/processed/'

    # plot_data_heatmap(input, channel_indices = [1,2], plot_together=True)

    selected_samples = ['E1_1', 'E1_2']
    plot_transmittance(input, selected_samples=selected_samples)

    # main()
    

