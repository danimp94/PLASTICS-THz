import numpy as np
import pandas as pd
import os


# Function to find the line number of the ***End_of_Header*** section
def find_header_end(file_path):
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            if "***End_of_Header***" in line:
                return line_num + 1  # Data starts immediately after the header
    return None


# Function to read LVM data (your custom function)
def read_lvm(file_path, start_line):
    data = []

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= start_line:  # Skip lines until reaching the data section
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


# Main function to process the file
def process_lvm_file(file_path, output_path, discard_percentage=0, discard_mode='first'):
    # Find the line number where data starts (after ***End_of_Header***)
    header_line = find_header_end(file_path)

    # Check if header was found
    if header_line is None:
        print("Error: Could not find the ***End_of_Header*** marker in the file.")
        return

    # Read the LVM data starting from the correct line
    data = read_lvm(file_path, header_line)

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

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data, columns=channel_names)

    # Discard data if percentage is greater than 0
    if discard_percentage > 0:
        df = discard_data(df, discard_percentage, discard_mode)

    # Write the DataFrame to the specified output file with semicolon separated values and no index
    df.to_csv(output_path, sep=';', index=False)


def discard_data(df, percentage, mode='last'):
    """
    Discards a percentage of data for each value of the 'Frequency' column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    percentage (float): The percentage of data to discard (0 < percentage < 100).
    mode (str): The mode of discarding data ('first', 'last', 'random').
    
    Returns:
    pd.DataFrame: The DataFrame after discarding the data.
    """
    if not 0 < percentage < 100:
        raise ValueError("Percentage must be between 0 and 100.")
    
    def discard_group(group):
        n = len(group)
        k = int(n * (percentage / 100))
        
        if mode == 'first':
            return group.iloc[k:]
        elif mode == 'last':
            return group.iloc[:-k]
        elif mode == 'random':
            return group.sample(frac=(1 - percentage / 100))
        else:
            raise ValueError("Mode must be 'first', 'last', or 'random'.")
    
    return df.groupby('Frequency (GHz)').apply(discard_group).reset_index(drop=True)



# Main execution block
if __name__ == "__main__":

    # Load the LVM file and define the output file path
    input_file_path = '../../data/experiment_1_plastics/test_03.1.lvm'

    # Define the output directory
    output_dir = '../../data/experiment_1_plastics/clean'

    # Create a "clean" directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file path
    output_file_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.lvm', '.csv'))

    # # Define the input directory
    # input_dir = '../../data/experiment_1_plastics'

    # # Loop through all .lvm files in the input directory
    # for filename in os.listdir(input_dir):
    #     if filename.endswith('.lvm'):
    #         input_file_path = os.path.join(input_dir, filename)
    #         output_file_path = os.path.join(output_dir, filename.replace('.lvm', '.csv'))

    #         # Call the main processing function for each file
    #         process_lvm_file(input_file_path, output_file_path)
    # Call the main processing function
    process_lvm_file(input_file_path, output_file_path, discard_percentage = 0, discard_mode='last')