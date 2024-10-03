import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_lvm(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Debugging: Print the line being read
            # print(f"Reading line: {line.strip()}")
            try:
                # Replace commas with periods and split by tab characters
                line = line.replace(',', '.').strip()  # Ensure any extra spaces are removed
                line_data = list(map(float, line.split('\t')))
                # print(f"Parsed line data: {line_data}")  # Debugging: Print parsed data
                if len(line_data) == 7:  # Expecting exactly 7 columns
                    data.append(line_data)  # Append all 7 columns
                else:
                    print(f"Skipping line with unexpected column count: {line_data}")
            except ValueError as e:
                print(f"Skipping line due to ValueError: {e}")
                continue

    return np.array(data) if data else np.array([])

def read_csv(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, delimiter=';')
        
        # Convert the DataFrame to a NumPy array
        data = df.to_numpy()
        
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return np.array([])

def plot_data_heatmap(data_files, channel_indices, plot_together=True):
    if plot_together:
        plt.figure(figsize=(16, 12))

    # Iterate through each file and plot the data
    for file_path in data_files:
        if not plot_together:
            plt.figure(figsize=(16, 12))

        data = read_csv(file_path)

        if data.size > 0:
            x_data = data[:, 0]  # Frequency (GHz)
            for index in channel_indices:
                if index < data.shape[1]:  # Ensure the index is valid
                    channel_data = data[:, index]
                    
                    # Filter out the x values for freq=210 and freq=600
                    mask = (x_data != 600) 
                    # mask = (x_data != 200) & (x_data != 210) & (x_data != 600)
                    x_data_filtered = x_data[mask]
                    channel_data_filtered = channel_data[mask]

                    label = f'Channel {index} from {os.path.basename(file_path)}'
                    
                    # Create a 2D histogram (heatmap)
                    heatmap, xedges, yedges = np.histogram2d(x_data_filtered, channel_data_filtered, bins=50)
                    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

                    plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
                    # plt.colorbar(label='Counts')

                else:
                    print(f"Invalid channel index: {index} for file {file_path}")

        if not plot_together:
            plt.title(f'Heatmap of LVM Data (Selected Channels vs Frequency) - {os.path.basename(file_path)}', fontsize=20)
            plt.xlabel('Frequency (GHz)', fontsize=16)
            plt.ylabel('Values (mV)', fontsize=16)
            plt.grid(True)

            # Save the plot
            save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Extract the base name of the current data file without extension
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(save_dir, f'heatmap_plot_{base_name}.png')
            plt.savefig(save_path, dpi=300)  # Set DPI to 300 for HD quality
            print(f"Heatmap plot saved as: {save_path}")
            # plt.show()

    if plot_together:
        plt.title('Heatmap of LVM Data (Selected Channels vs Frequency)', fontsize=20)
        plt.xlabel('Frequency (GHz)', fontsize=16)
        plt.ylabel('Values (mV)', fontsize=16)
        plt.grid(True)

        # Save the plot
        save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Extract the base name of the first data file without extension
        base_name = os.path.splitext(os.path.basename(data_files[0]))[0]
        save_path = os.path.join(save_dir, f'heatmap_plot_{base_name}_combined.png')
        plt.savefig(save_path, dpi=300)  # Set DPI to 300 for HD quality
        print(f"Combined heatmap plot saved as: {save_path}")
        #plt.show() 


def plot_data_overlay(data_files, channel_indices):
    plt.figure(figsize=(16, 12))

    # Iterate through each file and plot the data
    for file_path in data_files:
        data = read_csv(file_path)

        if data.size > 0:
            x_data = data[:, 0]  # Frequency (GHz)

            for index in channel_indices:
                if index < data.shape[1]:  # Ensure the index is valid
                    channel_data = data[:, index]
                    
                    # Filter out the x values for freq=210 and freq=600
                    mask = (x_data != 600) & (x_data != 210)
                    x_filtered = x_data[mask]
                    y_filtered = channel_data[mask]

                    # Plot the data for the current file
                    label = f'Channel {index} from {os.path.basename(file_path)}'
                    plt.scatter(x_filtered, y_filtered, label=label, alpha=0.6)

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Values (mV)')
    plt.title('Spectroscopy Sample Overlay (Scattered)')
    plt.legend()
    plt.show()

def plot_data_overlay_average(data_files, channel_indices):
    plt.figure(figsize=(16, 12))

    # Iterate through each file and plot the averaged data
    for file_path in data_files:
        data = read_csv(file_path)

        if data.size > 0:
            x_data = data[:, 0]  # Frequency (GHz)
            aggregated_data = {}

            for index in channel_indices:
                if index < data.shape[1]:  # Ensure the index is valid
                    channel_data = data[:, index]
                    
                    # Filter out the x values for freq=210 and freq=600
                    mask = mask = np.ones_like(x_data, dtype=bool)
                    # mask = (x_data != 600) & (x_data != 210)
                    x_filtered = x_data[mask]
                    y_filtered = channel_data[mask]

                    # Aggregate data
                    for x, y in zip(x_filtered, y_filtered):
                        if x not in aggregated_data:
                            aggregated_data[x] = []
                        aggregated_data[x].append(y)

            # Calculate the average for each frequency
            avg_data = {x: np.mean(y) for x, y in aggregated_data.items()}

            # Sort the data by frequency
            sorted_freqs = sorted(avg_data.keys())
            sorted_avg_values = [avg_data[freq] for freq in sorted_freqs]

            # Plot the averaged data for the current file
            label = f'Average from {os.path.basename(file_path)}'
            plt.plot(sorted_freqs, sorted_avg_values, label=label)

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Values (mV)')
    plt.title('Spectroscopy Sample Overlay (Averaged)')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # List of LVM files to be plotted
    processed_folder = '../../data/experiment_1_plastics/processed/'
    data_files = sorted([os.path.join(processed_folder, f) for f in os.listdir(processed_folder) if f.endswith('.csv')])
    data_files = data_files[2:4] + [data_files[-1]] # Select files 3 and 4

    # Check if all files exist
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            exit(1)  # Exit if any file is not found

    # Define the channels to plot (0 for Frequency, 1 for LG, 2 for HG, etc.)
    channels_to_plot = [1, 2]  # Adjust this list to select different channels

    # plot_data_heatmap(data_files, channels_to_plot, plot_together=False)

    # plot_data_overlay(data_files, channels_to_plot)
    plot_data_overlay_average(data_files, channels_to_plot)