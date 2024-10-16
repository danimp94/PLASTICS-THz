import numpy as np
import pandas as pd
import os
import argparse


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


# Main function to process the file
def process_lvm_file(file_path, output_path, discard_percentage=0, discard_mode='first'):
    # Find the line number where data starts (after ***End_of_Header***)
    header_line = find_header_end(file_path)

    # Check if header was found
    if (header_line is None):
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

def process_single_file(input_file_path, output_dir):
    output_file_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.lvm', '.csv'))
    process_lvm_file(input_file_path, output_file_path, discard_percentage=0, discard_mode='first')

def process_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith('.lvm'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename.replace('.lvm', '.csv'))
            process_lvm_file(input_file_path, output_file_path, discard_percentage=0, discard_mode='first')
  
def concatenate_csv_files(file1, file2, output_file):
    # Read the CSV files
    df1 = pd.read_csv(file1, delimiter=';')
    df2 = pd.read_csv(file2, delimiter=';')
    
    # Concatenate the DataFrames
    concatenated_df = pd.concat([df1, df2])
    
    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False, sep=';')

def merge_files(input_dir, output_file):
    all_data = []

    # Process each file in the input directory
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            print(f"Processing file: {file_path}")  # Debugging line
            try:
                df = pd.read_csv(file_path, delimiter=';')
                df.insert(0, 'Sample', os.path.splitext(file)[0])  # Add the new column with the filename (without .csv) at the beginning
                all_data.append(df)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    # Concatenate all DataFrames
    merged_df = pd.concat(all_data, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, sep=';', index=False)
    print(f"Merged file saved to: {output_file}")

def calculate_averages(input_file, output_file):
    try:
        # Read the merged data file
        df = pd.read_csv(input_file, delimiter=';')
        
        # Group by 'Sample' and 'Frequency (GHz)', then calculate mean and std deviation for 'HG (mV)' and 'LG (mV)'
        result = df.groupby(['Sample', 'Frequency (GHz)'])[['HG (mV)', 'LG (mV)']].agg(['mean', 'std']).reset_index()
        
        # Flatten the MultiIndex columns
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        
        # Merge the thickness values back into the result
        if 'Thickness (mm)' in df.columns:
            thickness_df = df[['Sample', 'Thickness (mm)']].drop_duplicates(subset=['Sample'])
            result = result.merge(thickness_df, on='Sample', how='left')
        
        # Save the result to a new CSV file
        result.to_csv(output_file, sep=';', index=False)
        print(f"Processed file saved to: {output_file}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def concatenate_pair_files(input_dir, output_dir):
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    for i in range(0, len(all_files), 2):
        if i + 1 < len(all_files):
            file1 = os.path.join(input_dir, all_files[i])
            file2 = os.path.join(input_dir, all_files[i + 1])
            base_name = all_files[i][:-6]
            concatenated_file = os.path.join(output_dir, f"{base_name}_{i//2}.csv")
            concatenate_csv_files(file1, file2, concatenated_file)
            

def remove_columns(input_dir, output_dir, num_columns, position='last', specific_positions=None):
    """
    Remove a specified number of columns from CSV files in a directory.
    Parameters:
    input_dir (str): Directory containing CSV files.
    output_dir (str): Directory to save the modified CSV files.
    num_columns (int): Number of columns to remove.
    position (str): Position of columns to remove ('first', 'last', or 'specific').
    specific_positions (list): List of specific column indices to remove (only used if position is 'specific').
    """
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, delimiter=';')
                if position == 'last':
                    df = df.iloc[:, :-num_columns]
                elif position == 'first':
                    df = df.iloc[:, num_columns:]
                elif position == 'specific':
                    if specific_positions is None:
                        raise ValueError("specific_positions must be provided when position is 'specific'.")
                    df = df.drop(df.columns[specific_positions], axis=1)
                else:
                    raise ValueError("Position must be 'first', 'last', or 'specific'.")
                output_file_path = os.path.join(output_dir, file)
                df.to_csv(output_file_path, sep=';', index=False)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

def add_thickness_column(input_dir, output_dir, characteristics_file):
    try:
        characteristics_df = pd.read_csv(characteristics_file, delimiter=';', encoding='utf-8-sig')
        characteristics_df.columns = characteristics_df.columns.str.strip()
        thickness_values = dict(zip(characteristics_df['sample'], characteristics_df['thickness']))
    except Exception as e:
        print(f"An error occurred while reading the characteristics file: {e}")
        return

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, delimiter=';')
                sample_name = os.path.splitext(file)[0]
                thickness_value = thickness_values.get(sample_name, None)
                if thickness_value is not None:
                    df['Thickness (mm)'] = thickness_value
                else:
                    print(f"No thickness value defined for file: {file}")
                output_file_path = os.path.join(output_dir, file)
                df.to_csv(output_file_path, sep=';', index=False)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

def calculate_transmittance(input_file, output_dir):
    try:
        # Read the data file
        df = pd.read_csv(input_file, delimiter=';')

        # Ensure the 'Sample' column is treated as a string
        df['Sample'] = df['Sample'].astype(str)

        # Separate the REF sample data
        ref_df = df[df['Sample'] == 'REF']

        # Merge the REF data back into the main DataFrame on 'Frequency (GHz)'
        merged_df = df.merge(ref_df, on='Frequency (GHz)', suffixes=('', '_REF'))

        # Calculate the transmittance for HG and LG mean values
        merged_df['HG (mV) mean Transmittance'] = merged_df['HG (mV) mean'] / merged_df['HG (mV) mean_REF']
        merged_df['LG (mV) mean Transmittance'] = merged_df['LG (mV) mean'] / merged_df['LG (mV) mean_REF']

        # Drop the REF columns used for calculation
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_REF')])

        # Construct the output file path
        output_file = os.path.join(output_dir, 'transmittance_results.csv')

        # Save the result to a new CSV file
        merged_df.to_csv(output_file, sep=';', index=False)
        print(f"Processed file saved to: {output_file}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")



def calculate_averages_and_dispersion(input_path, data_percentage=5, output_path=None):
    """
    For each frequency value, calculates the average, standard deviation, and variance of LG and HG
    within a preset window (%data) and saves it in another output file. Just one value per window.
    If input_path is a directory, processes all CSV files in the directory.
    """
    def process_file(input_file, output_file):
        df = pd.read_csv(input_file, delimiter=';')

        # Calculate the number of rows for each frequency
        freq_counts = df['Frequency (GHz)'].value_counts().sort_index()
        freq_counts = freq_counts.reset_index()
        freq_counts.columns = ['Frequency (GHz)', 'count'] 

        # Calculate the window size for each frequency as a percentage of the total rows for that frequency
        freq_counts['window_size'] = (freq_counts['count'] * data_percentage / 100).astype(int)

        results = []

        for _, row in freq_counts.iterrows():
            freq = row['Frequency (GHz)']
            window_size = int(row['window_size'])
            print(f"Processing frequency: {freq} with window size: {window_size}")
            
            # Ensure window_size is at least 1
            if window_size < 1:
                window_size = 1
            
            # Select the data for the current frequency
            freq_data = df[df['Frequency (GHz)'] == freq]
            
            # Iterate over the data in chunks of window_size
            for start in range(0, len(freq_data), window_size):
                window_data = freq_data.iloc[start:start + window_size]
                            
                # Calculate the mean, std deviation, and variance for LG and HG
                mean_values = window_data[['LG (mV)', 'HG (mV)']].mean()
                std_deviation_values = window_data[['LG (mV)', 'HG (mV)']].std()
                variance_values = window_data[['LG (mV)', 'HG (mV)']].var()
                
                # Append the results
                results.append({
                    'Frequency (GHz)': freq,
                    'LG (mV) mean': mean_values['LG (mV)'],
                    'HG (mV) mean': mean_values['HG (mV)'],
                    'LG (mV) std deviation': std_deviation_values['LG (mV)'],
                    'HG (mV) std deviation': std_deviation_values['HG (mV)'],
                    'LG (mV) variance': variance_values['LG (mV)'],
                    'HG (mV) variance': variance_values['HG (mV)']
                })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save the results to the output file
        results_df.to_csv(output_file, sep=';', index=False)
        print(f"Processed {input_file} and saved to {output_file}")

    def generate_output_filename(filename):
        base_name, ext = os.path.splitext(filename)
        parts = base_name.split('_')
        if len(parts) > 1:
            parts.insert(1, 'dispersion')
            new_base_name = '_'.join(parts)
        else:
            new_base_name = f"{base_name}_dispersion"
        return f"{new_base_name}{ext}"

    # Check if input_path is a directory
    if os.path.isdir(input_path):
        if output_path is None:
            raise ValueError("Output path must be specified when input is a directory.")
        
        # Ensure the output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Process each CSV file in the input directory
        for filename in os.listdir(input_path):
            if filename.endswith(".csv"):
                input_file = os.path.join(input_path, filename)
                output_file = os.path.join(output_path, generate_output_filename(filename))
                process_file(input_file, output_file)
    else:
        # Process a single file
        if output_path is None:
            output_file = generate_output_filename(os.path.basename(input_path))
            output_file = os.path.join(os.path.dirname(input_path), output_file)
        else:
            output_file = generate_output_filename(os.path.basename(output_path))
            output_file = os.path.join(os.path.dirname(output_path), output_file)
        process_file(input_path, output_file)


def main():
    parser = argparse.ArgumentParser(description="Process spectroscopy data.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for processing a single file
    parser_single = subparsers.add_parser("process_single", help="Process a single LVM file")
    parser_single.add_argument("input_file", help="Path to the input LVM file")
    parser_single.add_argument("output_dir", help="Directory to save the processed CSV file")

    # Subparser for processing multiple files
    parser_multiple = subparsers.add_parser("process_multiple", help="Process multiple LVM files")
    parser_multiple.add_argument("input_dir", help="Directory containing LVM files")
    parser_multiple.add_argument("output_dir", help="Directory to save the processed CSV files")

    # Subparser for concatenating files
    parser_concat = subparsers.add_parser("concatenate", help="Concatenate CSV files")
    parser_concat.add_argument("input_dir", help="Directory containing CSV files")
    parser_concat.add_argument("output_dir", help="Directory to save the concatenated CSV files")

    # Subparser for removing columns
    parser_remove = subparsers.add_parser("remove_columns", help="Remove columns from CSV files")
    parser_remove.add_argument("input_dir", help="Directory containing CSV files")
    parser_remove.add_argument("output_dir", help="Directory to save the modified CSV files")
    parser_remove.add_argument("num_columns", type=int, help="Number of columns to remove from the CSV files")
    parser_remove.add_argument("position", choices=['last', 'first', 'specific'], default='last', help="Position of columns to remove: 'last', 'first', or 'specific'")
    parser_remove.add_argument("--specific_columns", nargs='*', help="List of specific columns to remove (required if position is 'specific')")

    # Subparser for adding thickness column
    parser_thickness = subparsers.add_parser("add_thickness", help="Add thickness column to CSV files")
    parser_thickness.add_argument("input_dir", help="Directory containing CSV files")
    parser_thickness.add_argument("output_dir", help="Directory to save the modified CSV files")
    parser_thickness.add_argument("characteristics_file", help="Path to the characteristics CSV file")

    # Subparser for merging files
    parser_merge = subparsers.add_parser("merge", help="Merge all CSV files in a directory")
    parser_merge.add_argument("input_dir", help="Directory containing CSV files")
    parser_merge.add_argument("output_file", help="Path to save the merged CSV file")

    # Subparser for calculating averages and standard deviations
    parser_avg = subparsers.add_parser("calculate_averages", help="Calculate averages and standard deviations")
    parser_avg.add_argument("input_file", help="Path to the input CSV file")
    parser_avg.add_argument("output_file", help="Path to save the output CSV file")

    # Subparser for calculating transmittance
    parser_transmittance = subparsers.add_parser("calculate_transmittance", help="Calculate transmittance")
    parser_transmittance.add_argument("input_file", help="Path to the input CSV file")
    parser_transmittance.add_argument("output_dir", help="Path to save the output CSV file")

    # Subparser for calculating averages and dispersion
    parser_avg_disp = subparsers.add_parser("calculate_averages_and_dispersion", help="Calculate averages and dispersion")
    parser_avg_disp.add_argument("input_file", help="Path to the input CSV file")
    parser_avg_disp.add_argument("output_file", help="Path to save the output CSV file")
    parser_avg_disp.add_argument("--data_percentage", type=float, default=50, help="Percentage of data to consider for each frequency")

    args = parser.parse_args()

    if args.command == "process_single":
        process_single_file(args.input_file, args.output_dir)
    elif args.command == "process_multiple":
        process_files(args.input_dir, args.output_dir)
    elif args.command == "concatenate":
        concatenate_pair_files(args.input_dir, args.output_dir)
    elif args.command == "remove_columns":
        remove_columns(args.input_dir, args.output_dir, args.num_columns, args.position, args.specific_columns)
    elif args.command == "add_thickness":
        add_thickness_column(args.input_dir, args.output_dir, args.characteristics_file)
    elif args.command == "merge":
        merge_files(args.input_dir, args.output_file)
    elif args.command == "calculate_averages":
        calculate_averages(args.input_file, args.output_file)
    elif args.command == "calculate_transmittance":
        calculate_transmittance(args.input_file, args.output_dir)
    elif args.command == "calculate_averages_and_dispersion":
        calculate_averages_and_dispersion(args.input_file, args.data_percentage, args.output_file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

# # Input file path
#  ../../data/experiment_1_plastics/processed/merged_averages_std_dev.csv

# # Input directory
#  ../../data/experiment_1_plastics/processed/

# # Output file path
#  ../../data/experiment_1_plastics/processed/result/

# Example usage for every command (from /src/scripts directory):
# python preprocessing.py process_single ../../data/experiment_1_plastics/raw/sample1.lvm ../../data/experiment_1_plastics/processed/
# python preprocessing.py process_multiple ../../data/experiment_1_plastics/raw/ ../../data/experiment_1_plastics/processed/
# python preprocessing.py concatenate ../../data/experiment_1_plastics/processed_full/dispersion_2/ ../../data/experiment_1_plastics/processed_full/dispersion_2/conc/
# python preprocessing.py remove_columns ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/processed/ 4 last
# python preprocessing.py add_thickness ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/characteristics.csv
# python preprocessing.py merge ../../data/experiment_1_plastics/processed/ ../../data/experiment_1_plastics/processed/merged.csv
# python preprocessing.py calculate_averages ../../data/experiment_1_plastics/processed/merged.csv ../../data/experiment_1_plastics/processed/averages.csv
# python preprocessing.py calculate_transmittance ../../data/experiment_1_plastics/processed/averages.csv ../../data/experiment_1_plastics/processed/
# python preprocessing.py calculate_averages_and_dispersion ../../data/experiment_1_plastics/processed/averages.csv ../../data/experiment_1_plastics/processed/averages_dispersion.csv --data_percentage 50