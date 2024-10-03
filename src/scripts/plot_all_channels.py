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

def plot_data(data):
    if data.size == 0:
        print("No valid data to plot.")
        return

    if len(data.shape) != 2 or data.shape[1] != 7:
        print("Data format is incorrect. Expected 7 columns (7 data points per row).")
        return

    # Print range of values for each channel
    for i in range(7):
        print(f"Channel {i} - min: {data[:, i].min()}, max: {data[:, i].max()}")

    # Use row index as x-axis since 'time' is constant
    x_axis = np.arange(len(data))  # Artificial x-axis: 0, 1, 2, ..., len(data)

    print(f"Plotting data with shape: {data.shape}")
    print(f"x_axis length: {len(x_axis)}")

    # Define custom names for the channels
    channel_names = [
        "Frequency (GHz)",  # Channel 0
        "LG (mV)",        # Channel 1
        "HG (mV)",        # Channel 2
        "Current Laser 1 (mA)",        # Channel 3
        "Current Laser 2 (mA)",        # Channel 4
        "LD1 Laser point",        # Channel 5
        "LD2 Laser point"         # Channel 6
    ]

    # Plot all 7 channels with their names
    plt.figure(figsize=(12, 8))
    for i in range(0, 7):  # Plot channels from column 0 to 6
        plt.plot(x_axis, data[:, i], label=channel_names[i], marker='o', linestyle='-', markersize=4)

    plt.title('LVM Data Plot (7 Channels)')
    plt.xlabel('Sample Number')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)

    # Set plot limits for better visibility
    plt.xlim([0, len(x_axis)])
    # plt.ylim([
    #     min(data.min(axis=0)) - 10,  # Slightly below the minimum value
    #     max(data.max(axis=0)) + 10   # Slightly above the maximum value
    # ])

    plt.ylim([-1000, 1500])


    # Save plot in the same directory as the file
    save_dir = os.path.dirname(file_path)
    save_path = os.path.join(save_dir, "data_plot_02.png")
    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")
    plt.show()


if __name__ == "__main__":
    file_path = '../../test_2.0/test_02_10s_sweep_paper_test.lvm'
    data = read_lvm(file_path)

    if data.size > 0:
        print("Data successfully loaded.")
        print(f"Data shape: {data.shape}")
        print(data[:5])  # Print first 5 rows
    else:
        print("No data found or data could not be loaded.")

    plot_data(data)