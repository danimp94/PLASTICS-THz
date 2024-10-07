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
  
def concatenate_csv_files(file1, file2, output_file):
    # Read the CSV files
    df1 = pd.read_csv(file1, delimiter=';')
    df2 = pd.read_csv(file2, delimiter=';')
    
    # Concatenate the DataFrames
    concatenated_df = pd.concat([df1, df2])
    
    # Write the concatenated DataFrame to a new CSV file
    concatenated_df.to_csv(output_file, index=False, sep=';')

def merge_files_with_filename(input_dir, output_file):
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

def average_and_std_deviation(input_file, output_file):
    try:
        # Read the merged data file
        df = pd.read_csv(input_file, delimiter=';')
        
        # Group by 'Sample' and 'Frequency (GHz)', then calculate mean and std deviation for 'HG (mV)' and 'LG (mV)'
        result = df.groupby(['Sample', 'Frequency (GHz)'])[['HG (mV)', 'LG (mV)']].agg(['mean', 'std']).reset_index()
        if 'Thickness (mm)' in df.columns:
            thickness_df = df[['Sample', 'Thickness (mm)']].drop_duplicates(subset=['Sample'])
            result = result.merge(thickness_df, on='Sample', how='right')
        
        # Flatten the MultiIndex columns
        result.columns = [' '.join(col).strip() for col in result.columns.values]
        
        # Merge the thickness values back into the result
        if 'Thickness (mm)' in df.columns:
            thickness_df = df[['Sample', 'Thickness (mm)']].drop_duplicates(subset=['Sample'])
            result = result.merge(thickness_df, on='Sample', how='right')
        
        # Save the result to a new CSV file
        result.to_csv(output_file, sep=';', index=False)
        print(f"Processed file saved to: {output_file}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Main execution block
if __name__ == "__main__":

    # Define the output directory
    # output_dir = '../../data/experiment_1_plastics/processed/conc'

    # # Create a "processed" directory if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    # # SINGLE FILE PROCESSING
    # # Load the LVM file and define the output file path
    # input_file_path = '../../data/experiment_1_plastics/test_03.1.lvm'

    # # Define the output file path
    # output_file_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.lvm', '.csv'))

    # # Call the main processing function
    # process_lvm_file(input_file_path, output_file_path, discard_percentage = 50, discard_mode='first')

    # # Define the input directory
    # input_dir = '../../data/experiment_1_plastics/raw'

    # # # FILE PROCESSING (Discarding and converting to CSV)
    # # Loop through all .lvm files in the input directory
    # for filename in os.listdir(input_dir):
    #     if filename.endswith('.lvm'):
    #         input_file_path = os.path.join(input_dir, filename)
    #         output_file_path = os.path.join(output_dir, filename.replace('.lvm', '.csv'))

    #         # Call the main processing function for each file
    #         process_lvm_file(input_file_path, output_file_path, discard_percentage = 75, discard_mode='first')


    # # FILE CONTATENATION
    # input_dir = '../../data/experiment_1_plastics/processed'

    # # Get all CSV files in the input directory
    # all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])

    # # Loop through the files in pairs
    # for i in range(0, len(all_files), 2):
    #     if i + 1 < len(all_files):
    #         file1 = os.path.join(input_dir, all_files[i])
    #         file2 = os.path.join(input_dir, all_files[i + 1])
    #         base_name = all_files[i][:-6]  # Remove '.csv' and the last digit
    #         concatenated_file = os.path.join(output_dir, f"{base_name}_{i//2}.csv")

    #         # Concatenate the CSV files
    #         concatenate_csv_files(file1, file2, concatenated_file)

    # # REMOVE THE LAST 4 COLUMNS FOR EACH FILE
    # input_dir = '../../data/experiment_1_plastics/processed/'

    # for file in os.listdir(input_dir):
    #     if file.endswith('.csv'):
    #         file_path = os.path.join(input_dir, file)
    #         print(f"Processing file: {file_path}")  # Debugging line
    #         try:
    #             df = pd.read_csv(file_path, delimiter=';')
    #             df = df.iloc[:, :-4]  # Remove the last 4 columns
    #             output_file_path = os.path.join(output_dir, file)
    #             df.to_csv(output_file_path, sep=';', index=False)
    #         except FileNotFoundError as e:
    #             print(f"File not found: {e}")
    #         except Exception as e:
    #             print(f"An error occurred: {e}")


    # # # ADD A NEW COLUMN WITH EXTRA VALUES
    # input_dir = '../../data/experiment_1_plastics/processed/'
    # output_dir = '../../data/experiment_1_plastics/processed/conc'
    # characteristics_file = '../../data/experiment_1_plastics/characteristics.csv'

    # # Read the characteristics file to get thickness values
    # try:
    #     characteristics_df = pd.read_csv(characteristics_file, delimiter=';', encoding='utf-8-sig')  # Handle BOM and specify correct delimiter
    #     print("Column names in characteristics file:", characteristics_df.columns)  # Debugging line to print column names
    #     characteristics_df.columns = characteristics_df.columns.str.strip()  # Remove any leading/trailing spaces
    #     thickness_values = dict(zip(characteristics_df['sample'], characteristics_df['thickness']))
    # except Exception as e:
    #     print(f"An error occurred while reading the characteristics file: {e}")
    #     exit(1)

    # # Process each file in the input directory
    # for file in os.listdir(input_dir):
    #     if file.endswith('.csv'):
    #         file_path = os.path.join(input_dir, file)
    #         print(f"Processing file: {file_path}")  # Debugging line
    #         try:
    #             df = pd.read_csv(file_path, delimiter=';')
    #             sample_name = os.path.splitext(file)[0]  # Assuming the file name corresponds to the sample name
    #             thickness_value = thickness_values.get(sample_name, None)
    #             if thickness_value is not None:
    #                 df['Thickness (mm)'] = thickness_value  # Add the new column with the thickness value
    #             else:
    #                 print(f"No thickness value defined for file: {file}")
    #             output_file_path = os.path.join(output_dir, file)
    #             df.to_csv(output_file_path, sep=';', index=False)
    #         except FileNotFoundError as e:
    #             print(f"File not found: {e}")
    #         except Exception as e:
    #             print(f"An error occurred: {e}")


    # # MERGE ALL FILES IN THE INPUT DIRECTORY
    # input_dir = '../../data/experiment_1_plastics/processed/'
    # output_dir = '../../data/experiment_1_plastics/processed/'

    # # Merge all files in the input directory
    # merge_files_with_filename(input_dir, os.path.join(output_dir, 'merged_data.csv'))



    # # CALCULATE THE AVERAGE AND STANDARD DEVIATION FOR EACH FREQUENCY
    input_file = '../../data/experiment_1_plastics/processed/merged_data.csv'
    output_file = '../../data/experiment_1_plastics/processed/merged_averages_std_dev.csv'

    # Calculate the average and standard deviation for each frequency

    average_and_std_deviation(input_file, output_file)
