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

def plot_data_line(data, file_path):
    if data.size == 0:
        print("No valid data to plot.")
        return

    if len(data.shape) != 2 or data.shape[1] != 7:
        print("Data format is incorrect. Expected 7 columns (7 data points per row).")
        return

    # Print range of values for each channel
    for i in range(7):
        print(f"Channel {i} - min: {data[:, i].min()}, max: {data[:, i].max()}")

    # Define custom names for the channels
    channel_names = [
        "Frequency (GHz)",  # Channel 0
        "LG (mV)",          # Channel 1
        "HG (mV)",          # Channel 2
        "Current Laser 1 (mA)",  # Channel 3
        "Current Laser 2 (mA)",  # Channel 4
        "LD1 Laser point",      # Channel 5
        "LD2 Laser point"       # Channel 6
    ]

    # Extract x and y data
    x_data = data[:, 0]  # Frequency (GHz)
    lg_data = data[:, 1] # LG (mV)
    # hg_data = data[:, 2] # HG (mV)

    # Plot LG and HG vs Frequency
    plt.figure(figsize=(12, 8))

    plt.plot(x_data, lg_data, label='LG (mV)', marker='o', linestyle='-', markersize=4)
    # plt.plot(x_data, hg_data, label='HG (mV)', marker='x', linestyle='--', markersize=4)

    plt.title('LVM Data Plot (LG and HG vs Frequency)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Values (mV)')
    plt.legend()
    plt.grid(True)

    # Set plot limits for better visibility
    # plt.xlim([x_data.min(), x_data.max()])
    # plt.ylim([
    #     lg_data.min(),  # Slightly below the minimum value
    #     lg_data.max()  # Slightly above the maximum value
    # ])

    plt.ylim([ -1000, 1500 ])

    # Create a unique filename with a numerical suffix
    save_dir = os.path.dirname(file_path,'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_filename = "plot"
    suffix = 1
    while True:
        save_path = os.path.join(save_dir, f"{base_filename}_{suffix}.png")
        if not os.path.exists(save_path):
            break
        suffix += 1

    plt.savefig(save_path)
    print(f"Plot saved as: {save_path}")
    plt.show()



def plot_data(data, file_path):
    if data.size == 0:
        print("No valid data to plot.")
        return

    if len(data.shape) != 2 or data.shape[1] != 7:
        print("Data format is incorrect. Expected 7 columns (7 data points per row).")
        return

    # Print range of values for each channel
    for i in range(7):
        print(f"Channel {i} - min: {data[:, i].min()}, max: {data[:, i].max()}")

    # Define custom names for the channels
    channel_names = [
        "Frequency (GHz)",  # Channel 0
        "LG (mV)",  # Channel 1
        "HG (mV)",  # Channel 2
        "Current Laser 1 (mA)",  # Channel 3
        "Current Laser 2 (mA)",  # Channel 4
        "LD1 Laser point",  # Channel 5
        "LD2 Laser point"  # Channel 6
    ]

    # Extract x and y data
    x_data = data[:, 0]  # Frequency (GHz)
    lg_data = data[:, 1]  # LG (mV)
    hg_data = data[:, 2]  # HG (mV)

    # Plot LG and HG vs Frequency as scatter plots
    plt.figure(figsize=(12, 8))

    plt.scatter(x_data, lg_data, label='LG (mV)', color='blue', marker='o', s=10, alpha=0.6)
    plt.scatter(x_data, hg_data, label='HG (mV)', color='red', marker='x', s=10, alpha=0.6)

    plt.title('LVM Data Scatter Plot (LG and HG vs Frequency)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Values (mV)')
    plt.legend()
    plt.grid(True)

    # Set plot limits for better visibility
    plt.xlim([x_data.min(), x_data.max()])
    # plt.ylim([
    #     min(lg_data.min(), hg_data.min()) - 10,  # Slightly below the minimum value
    #     max(lg_data.max(), hg_data.max()) + 10  # Slightly above the maximum value
    # ])

    plt.ylim([-1000, 1500])


    # Create a unique filename with a numerical suffix
    save_dir = os.path.join(os.path.dirname(file_path),'plots')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_filename = "scatter_plot"
    suffix = 1
    while True:
        save_path = os.path.join(save_dir, f"{base_filename}_{suffix}.png")
        if not os.path.exists(save_path):
            break
        suffix += 1

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

    plot_data(data, file_path)

