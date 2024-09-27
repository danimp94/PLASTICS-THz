import os
import numpy as np
import pandas as pd


# Function to find the line number of the ***End_of_Header*** section
def find_header_end(file_path):
    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):
            if "***End_of_Header***" in line:
                return line_num + 1  # Data starts immediately after the header


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


# Load the LVM file and identify where the data starts
file_path = '../../data/experiment_1_plastics/test_04.1_test.lvm'

# Find the line number where data starts (after ***End_of_Header***)
header_line = find_header_end(file_path)

# Check if header was found
if header_line is None:
    print("Error: Could not find the ***End_of_Header*** marker in the file.")
else:
    # Read the LVM data starting from the correct line
    data = read_lvm(file_path, header_line)

    # If data was successfully read, save it to a new file
    if data.size > 0:
        # Convert to a pandas DataFrame
        df = pd.DataFrame(data, columns=['Column1',
                                         'Column2',
                                         'Column3',
                                         'Column4',
                                         'Column5',
                                         'Column6',
                                         'Column7'])

        # Save the data with headers to a new LVM file, appending "_proc" to the filename
        new_file_path = os.path.splitext(file_path)[0] + '_proc.lvm'

        # Write the DataFrame to a new file with tab-separated values and no index
        df.to_csv(new_file_path, sep='\t', index=False)

        print(f"Data with headers saved to {new_file_path}")
    else:
        print("No data found to save.")