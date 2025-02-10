import numpy as np
import pandas as pd
import os
import argparse

def find_header_end(file_path):
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file):
                if "***End_of_Header***" in line:
                    return line_num + 1  # Data starts immediately after the header
    except Exception as e:
        print(f"End of header not found: {e}")
    return None

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

def process_file_to_csv(file_path, output_path, discard_first_percentage=0, discard_last_percentage=0, channel_names=None):
    if file_path.endswith('.lvm'):
        header_line = find_header_end(file_path)
        if header_line is None:
            print("Error: Could not find the ***End_of_Header*** marker in the file.")
            #return
            header_line = 0
        data = read_lvm(file_path, header_line)
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path, delimiter=';').values
    else:
        print("Error: Unsupported file format. Only .lvm and .csv files are supported.")
        return

    if data.size == 0:
        print("Error: No data found in the file.")
        return

    if channel_names is None:
        num_columns = data.shape[1]
        column_names = [f"Column {i}" for i in range(num_columns)]
    else:
        column_names = channel_names

    if discard_first_percentage + discard_last_percentage >= 100:
        raise ValueError("The sum of discard_first_percentage and discard_last_percentage must be less than 100.")

    df = pd.DataFrame(data, columns=column_names)

    if discard_first_percentage > 0:
        df = discard_data(df, discard_first_percentage, 0)
    if discard_last_percentage > 0:
        df = discard_data(df, 0, discard_last_percentage)

    df.to_csv(output_path, sep=';', index=False)


def discard_data(df, discard_first_percentage, discard_last_percentage):
    def discard_group(group):
        frequency = group['Frequency (GHz)'].iloc[0]
        if frequency <= 200.0: # Percentage of data removed for frequencies <= 200.0 GHz
            first_percentage = discard_first_percentage
        else: # Percentage of data removed for frequencies > 200.0 GHz
            first_percentage = discard_first_percentage * 0.74  

        n = len(group)
        k_first = int(n * (first_percentage / 100))
        k_last = int(n * (discard_last_percentage / 100))

        return group.iloc[k_first:n - k_last]

    return df.groupby('Frequency (GHz)').apply(discard_group).reset_index(drop=True)


def process_files(input_path, output_dir, discard_first_percentage=0, discard_last_percentage=0,  channel_names=None):
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)
    
    print(f"Input path: {input_path}")  # Debugging line
    print(f"Output directory: {output_dir}")  # Debugging line
   
    if os.path.isfile(input_path):
        output_file_path = os.path.join(output_dir, os.path.basename(input_path).replace('.lvm', '.csv'))
        process_file_to_csv(input_path, output_file_path, discard_first_percentage, discard_last_percentage, channel_names)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith('.lvm') or filename.endswith('.csv'):
                input_file_path = os.path.join(input_path, filename)
                output_file_path = os.path.join(output_dir, filename.replace('.lvm', '.csv'))
                process_file_to_csv(input_file_path, output_file_path, discard_first_percentage, discard_last_percentage, channel_names)                

    else:
        raise ValueError("Input path must be a file or directory.")

def concatenate_csv_files(file1, file2, output_file):
    df1 = pd.read_csv(file1, delimiter=';')
    df2 = pd.read_csv(file2, delimiter=';')
    concatenated_df = pd.concat([df1, df2])
    concatenated_df.to_csv(output_file, index=False, sep=';')

def concatenate_pair_files(input_dir, output_dir):
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    for i in range(0, len(all_files), 2):
        if i + 1 < len(all_files):
            file1 = os.path.join(input_dir, all_files[i])
            file2 = os.path.join(input_dir, all_files[i + 1])
            base_name = all_files[i][:-6]
            concatenated_file = os.path.join(output_dir, f"{base_name}_{i//2}.csv")
            concatenate_csv_files(file1, file2, concatenated_file)

## Merge all CSV files in a directory
def merge_files(input_dir, output_file):
    all_data = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, delimiter=';')
                all_data.append(df)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(output_file, sep=';', index=False)
    print(f"Merged file saved to: {output_file}")

## Add sample name column to the data
def add_sample_name_column(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(input_dir, file)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path, delimiter=';')
                df.insert(0, 'Sample', os.path.splitext(file)[0])
                output_file_path = os.path.join(output_dir, file)
                df.to_csv(output_file_path, sep=';', index=False)
            except FileNotFoundError as e:
                print(f"File not found: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

def calculate_averages(input_file, output_file):
    try:
        df = pd.read_csv(input_file, delimiter=';')
        result = df.groupby(['Sample', 'Frequency (GHz)'])[['HG (mV)', 'LG (mV)']].agg(['mean', 'std']).reset_index()
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        if 'Thickness (mm)' in df.columns:
            thickness_df = df[['Sample', 'Thickness (mm)']].drop_duplicates(subset=['Sample'])
            result = result.merge(thickness_df, on='Sample', how='left')
        result.to_csv(output_file, sep=';', index=False)
        print(f"Processed file saved to: {output_file}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def remove_columns(input_dir, output_dir, num_columns, position='last', specific_positions=None):
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
        df = pd.read_csv(input_file, delimiter=';')
        df['Sample'] = df['Sample'].astype(str)
        ref_df = df[df['Sample'] == 'REF']
        merged_df = df.merge(ref_df, on='Frequency (GHz)', suffixes=('', '_REF'))
        merged_df['HG (mV) mean Transmittance'] = merged_df['HG (mV) mean'] / merged_df['HG (mV) mean_REF']
        merged_df['LG (mV) mean Transmittance'] = merged_df['LG (mV) mean'] / merged_df['LG (mV) mean_REF']
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_REF')])
        output_file = os.path.join(output_dir, 'transmittance_results.csv')
        merged_df.to_csv(output_file, sep=';', index=False)
        print(f"Processed file saved to: {output_file}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_averages_and_dispersion(input_path, data_percentage=5, output_path=None):
    def process_file(input_file, output_file):
        df = pd.read_csv(input_file, delimiter=';')
        results = []
        for (sample, freq), group in df.groupby(['Sample', 'Frequency (GHz)']):
            window_size = max(1, int(len(group) * data_percentage / 100))
            print(f"Processing sample: {sample}, frequency: {freq} with window size: {window_size}")
            for start in range(0, len(group), window_size):
                window_data = group.iloc[start:start + window_size]
                mean_values = window_data[['LG (mV)', 'HG (mV)']].mean()
                std_deviation_values = window_data[['LG (mV)', 'HG (mV)']].std()
                variance_values = window_data[['LG (mV)', 'HG (mV)']].var()
                results.append({
                    'Sample': sample,
                    'Frequency (GHz)': freq,
                    'LG (mV) mean': mean_values['LG (mV)'],
                    'HG (mV) mean': mean_values['HG (mV)'],
                    'LG (mV) std deviation': std_deviation_values['LG (mV)'],
                    'HG (mV) std deviation': std_deviation_values['HG (mV)'],
                    'LG (mV) variance': variance_values['LG (mV)'],
                    'HG (mV) variance': variance_values['HG (mV)'],
                    'Thickness (mm)': window_data['Thickness (mm)'].iloc[0]
                })
        results_df = pd.DataFrame(results)
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

    if os.path.isdir(input_path):
        if output_path is None:
            raise ValueError("Output path must be specified when input is a directory.")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename in os.listdir(input_path):
            if filename.endswith(".csv"):
                input_file = os.path.join(input_path, filename)
                output_file = os.path.join(output_path, generate_output_filename(filename))
                process_file(input_file, output_file)
    else:
        if output_path is None:
            output_file = generate_output_filename(os.path.basename(input_path))
            output_file = os.path.join(os.path.dirname(input_path), output_file)
        else:
            output_file = generate_output_filename(os.path.basename(output_path))
            output_file = os.path.join(os.path.dirname(output_path), output_file)
        process_file(input_path, output_file)


def main():
    return
#     parser = argparse.ArgumentParser(description="Process spectroscopy data.")
#     subparsers = parser.add_subparsers(dest="command")

#     # Subparser for processing files or directories
#     parser_process = subparsers.add_parser("process", help="Process LVM files or directories")
#     parser_process.add_argument("input_path", help="Path to the input LVM file or directory")
#     parser_process.add_argument("output_path", help="Path to save the processed CSV file or directory")
#     parser_process.add_argument("--discard_first_percentage", type=float, default=0, help="Percentage of data to discard from the beginning")
#     parser_process.add_argument("--discard_last_percentage", type=float, default=0, help="Percentage of data to discard from the end")
#     parser_process.add_argument("--discard_mode", choices=['first', 'last', 'random'], default='first', help="Mode of discarding data")
#     parser_process.add_argument("--channel_names", nargs='*', help="List of channel names")

#     # Subparser for concatenating files
#     parser_concat = subparsers.add_parser("concatenate", help="Concatenate CSV files")
#     parser_concat.add_argument("input_dir", help="Directory containing CSV files")
#     parser_concat.add_argument("output_dir", help="Directory to save the concatenated CSV files")

#     # Subparser for removing columns
#     parser_remove = subparsers.add_parser("remove_columns", help="Remove columns from CSV files")
#     parser_remove.add_argument("input_dir", help="Directory containing CSV files")
#     parser_remove.add_argument("output_dir", help="Directory to save the modified CSV files")
#     parser_remove.add_argument("num_columns", type=int, help="Number of columns to remove from the CSV files")
#     parser_remove.add_argument("position", choices=['last', 'first', 'specific'], default='last', help="Position of columns to remove: 'last', 'first', or 'specific'")
#     parser_remove.add_argument("--specific_columns", nargs='*', help="List of specific columns to remove (required if position is 'specific')")

#     # Subparser for adding thickness column
#     parser_thickness = subparsers.add_parser("add_thickness", help="Add thickness column to CSV files")
#     parser_thickness.add_argument("input_dir", help="Directory containing CSV files")
#     parser_thickness.add_argument("output_dir", help="Directory to save the modified CSV files")
#     parser_thickness.add_argument("characteristics_file", help="Path to the characteristics CSV file")

#     # Subparser for merging files
#     parser_merge = subparsers.add_parser("merge", help="Merge all CSV files in a directory")
#     parser_merge.add_argument("input_dir", help="Directory containing CSV files")
#     parser_merge.add_argument("output_file", help="Path to save the merged CSV file")

#     # Subparser for calculating averages and standard deviations
#     parser_avg = subparsers.add_parser("calculate_averages", help="Calculate averages and standard deviations")
#     parser_avg.add_argument("input_file", help="Path to the input CSV file")
#     parser_avg.add_argument("output_file", help="Path to save the output CSV file")

#     # Subparser for calculating transmittance
#     parser_transmittance = subparsers.add_parser("calculate_transmittance", help="Calculate transmittance")
#     parser_transmittance.add_argument("input_file", help="Path to the input CSV file")
#     parser_transmittance.add_argument("output_dir", help="Path to save the output CSV file")

#     # Subparser for calculating averages and dispersion
#     parser_avg_disp = subparsers.add_parser("calculate_averages_and_dispersion", help="Calculate averages and dispersion")
#     parser_avg_disp.add_argument("input_file", help="Path to the input CSV file")
#     parser_avg_disp.add_argument("output_file", help="Path to save the output CSV file")
#     parser_avg_disp.add_argument("--data_percentage", type=float, default=50, help="Percentage of data to consider for each frequency")

#     args = parser.parse_args()

#     if args.command == "process":
#         process_files(args.input_path, args.output_path, args.discard_first_percentage, args.discard_last_percentage,  args.discard_mode, args.channel_names)
#     elif args.command == "concatenate":
#         concatenate_pair_files(args.input_dir, args.output_dir)
#     elif args.command == "remove_columns":
#         remove_columns(args.input_dir, args.output_dir, args.num_columns, args.position, args.specific_columns)
#     elif args.command == "add_thickness":
#         add_thickness_column(args.input_dir, args.output_dir, args.characteristics_file)
#     elif args.command == "merge":
#         merge_files(args.input_dir, args.output_file)
#     elif args.command == "calculate_averages":
#         calculate_averages(args.input_file, args.output_file)
#     elif args.command == "calculate_transmittance":
#         calculate_transmittance(args.input_file, args.output_dir)
#     elif args.command == "calculate_averages_and_dispersion":
#         calculate_averages_and_dispersion(args.input_file, args.data_percentage, args.output_file)
#     else:
#         parser.print_help()

if __name__ == "__main__":

    # Processing pipeline (from /src/scripts directory):
    # Input file path
    input = "../../data/experiment_5_plastics/tmp/"
    input = "../../data/experiment_5_plastics/processed/tmp/"

    # input = "../../data/experiment_3_repeatibility/processed/test/"
    # input = "../../data/experiment_3_repeatibility/raw/"
    # input = "../../data/experiment_4_plastics/processed/new_sample/tmp/"



    # # # Output directory
    output = "../../data/experiment_5_plastics/processed/tmp/"
    output = "../../data/experiment_5_plastics/processed/new_sample/"
    # output = "../../data/experiment_3_repeatibility/processed/"


    # # Process files
    # channel_names = ["Frequency (GHz)", "LG (mV)", "HG (mV)"]
    # discard_first_percentage = 77 # Discard 77% of data from the beginning (50s/65s) *77x0.74 for frequencies > 200 GHz (20s/35s)
    # discard_last_percentage = 20 # Discard 20% of resulting data from the end after discarding 77% from the beginning (3s/15s)

    # process_files(input, output, discard_first_percentage, discard_last_percentage, channel_names)


    # Add sample name column
    add_sample_name_column(input, output)


    # Merge files
    # merge_files(input, os.path.join(output, 'test_merged_v1.csv'))

    # In Sample column, if starts with E --> E1, if starts with H --> H1, if starts with R --> REF
    # df = pd.read_csv(os.path.join(output, 'test_merged.csv'), delimiter=';')
    # df['Sample'] = df['Sample'].apply(lambda x: 'E1' if x.startswith('E') else 'H1' if x.startswith('H') else 'REF')
    # df.to_csv(os.path.join(output, 'test_merged_v1_pca.csv'), sep=';', index=False)
    



    # Calculate averages and dispersion for each frequency
    # calculate_averages_and_dispersion(input, data_percentage=100, output_path=os.path.join(output, 'dispersion.csv'))

    # time_window = 1 # Time window in seconds
    # data_percentage = time_window*100/12
    # calculate_averages_and_dispersion(output, data_percentage, output_path=os.path.join(output, 'dispersion'))


    # main()



