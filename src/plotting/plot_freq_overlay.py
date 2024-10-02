import os
import numpy as np
import matplotlib.pyplot as plt


def read_lvm(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Debugging: Print the line being read
            print(f"Reading line: {line.strip()}")
            try:
                # Replace commas with periods and split by tab characters
                line = line.replace(',', '.').strip()  # Ensure any extra spaces are removed
                line_data = list(map(float, line.split('\t')))
                print(f"Parsed line data: {line_data}")  # Debugging: Print parsed data
                if len(line_data) == 7:  # Expecting exactly 7 columns
                    data.append(line_data)  # Append all 7 columns
                else:
                    print(f"Skipping line with unexpected column count: {line_data}")
            except ValueError as e:
                print(f"Skipping line due to ValueError: {e}")
                continue

    return np.array(data) if data else np.array([])

def plot_data_overlayed(data_files, channel_indices):
    plt.figure(figsize=(16, 12))

    # Iterate through each file and plot the data
    for file_path in data_files:
        data = read_lvm(file_path)

        if data.size > 0:
            x_data = data[:, 0]  # Frequency (GHz)
            for index in channel_indices:
                if index < data.shape[1]:  # Ensure the index is valid
                    channel_data = data[:, index]
                    label = f'Channel {index} from {os.path.basename(file_path)}'
                    marker = 'o' if index % 2 == 0 else 'x'  # Alternate markers for better visibility
                    plt.scatter(x_data, channel_data, label=label, marker=marker, alpha=0.5, s=30)

                else:
                    print(f"Invalid channel index: {index} for file {file_path}")

    plt.title('Overlay of LVM Data (Selected Channels vs Frequency)', fontsize=20)  # Increased title font size
    plt.xlabel('Frequency (GHz)', fontsize=16)  # Increased xlabel font size
    plt.ylabel('Values (mV)', fontsize=16)  # Increased ylabel font size
    plt.legend(fontsize=14)  # Increased legend font size
    plt.grid(True)

    # Set y-axis limits
    plt.ylim([-1000, 1500])

    # Save the plot
    save_dir = os.path.join(os.path.dirname(data_files[0]), 'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'overlay_scatter_plot.png')
    plt.savefig(save_path, dpi=300)  # Set DPI to 300 for HD quality
    plt.savefig(save_path)
    print(f"Overlay scatter plot saved as: {save_path}")
    plt.show()

if __name__ == "__main__":
    # List of LVM files to be plotted
    data_files = [
        '../../data/experiment_1_plastics/clean/test_01.2_proc.lvm',
        '../../data/experiment_1_plastics/clean/test_02.2_proc.lvm',
        '../../data/experiment_1_plastics/clean/test_03.2_proc.lvm'
    ]

    # Check if all files exist
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            exit(1)  # Exit if any file is not found

    # Define the channels to plot (0 for Frequency, 1 for LG, 2 for HG, etc.)
    channels_to_plot = [1, 2]  # Adjust this list to select different channels

    plot_data_overlayed(data_files, channels_to_plot)