import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq 
from scipy.signal import detrend
import matplotlib.pyplot as plt

def fft_analysis(predefined_frequency): #NOT WORKING
    # Read csv file
    df = pd.read_csv('../../data/experiment_2_plastics/processed/E4_1.csv', delimiter=';')

    # Extract the signal from the dataframe
    signal_columns = ['LG (mV)', 'HG (mV)']
    # Filter the dataframe to include only rows where 'Frequency (GHz)' is equal to predefined_frequency
    filtered_df = df[df['Frequency (GHz)'] == predefined_frequency]
    
    # Extract the signal from the filtered dataframe
    signal = filtered_df[signal_columns[1]].values

    # Generate time array assuming 12 seconds of sampling
    num_samples = filtered_df.shape[0]
    
    # Count the number of values in the 'Frequency (GHZ)' column that are equal to the predefined frequency
    frequency_count = filtered_df.shape[0]
    print(f"Number of values in 'Frequency (GHZ)' column equal to {predefined_frequency}: {frequency_count}")
    t = np.linspace(0, 12, num_samples, endpoint=False)
    print(f"Number of samples: {num_samples}")

    sample_rate = 1 / (t[1] - t[0])  # Assuming uniform sampling
    print(f"Sample rate: {sample_rate} Hz")

    # Perform FFT
    yf = fft(signal)
    xf = fftfreq((len(t)), 1 / sample_rate)

    # Plot the original signal
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')

    # Plot the FFT result
    plt.subplot(2, 1, 2)
    plt.plot(xf, np.abs(yf))
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def analyze_spectrum(predefined_frequency): #NOT WORKING
    # Read csv file
    df = pd.read_csv('../../data/experiment_2_plastics/processed/E4_1.csv', delimiter=';')

    # Extract the signal from the dataframe
    signal_columns = ['LG (mV)', 'HG (mV)']
    # Filter the dataframe to include only rows where 'Frequency (GHz)' is equal to predefined_frequency
    filtered_df = df[df['Frequency (GHz)'] == predefined_frequency]
    
    # Extract the signal from the filtered dataframe
    signal = filtered_df[signal_columns[1]].values

    # Ensure the signal length matches the number of samples
    num_samples = filtered_df.shape[0]
    if num_samples == 0:
        print(f"No data available for frequency {predefined_frequency} GHz")
        return

    # Detrend the signal to remove DC component
    # signal = detrend(signal)

    # Generate time array assuming 12 seconds of sampling
    t = np.linspace(0, 12, num_samples, endpoint=False)

    # Convert sample rate from GHz to Hz
    sample_rate = predefined_frequency * 1e9  # Convert GHz to Hz

    # Perform FFT
    yf = fft(signal)
    xf = fftfreq(num_samples, 1 / sample_rate)

    # Convert FFT frequencies to GHz
    xf_ghz = xf / 1e9

    # Identify the peak frequency
    peak_frequency_index = np.argmax(np.abs(yf))
    peak_frequency = xf_ghz[peak_frequency_index]
    print(f"Peak frequency: {peak_frequency} GHz")

    # Check if the peak frequency matches the predefined frequency
    if np.isclose(peak_frequency, predefined_frequency, atol=0.01):
        print(f"The peak frequency {peak_frequency} GHz matches the predefined frequency {predefined_frequency} GHz")
    else:
        print(f"The peak frequency {peak_frequency} GHz does not match the predefined frequency {predefined_frequency} GHz")

    # Plot the original signal
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title(f'Original Signal at {predefined_frequency} GHz')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')

    # Plot the FFT result
    plt.subplot(2, 1, 2)
    plt.plot(xf_ghz, np.abs(yf) / num_samples)
    plt.title(f'FFT Spectrum at {predefined_frequency} GHz')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predefined_frequency = 240 # Replace with the actual predefined frequency value
    analyze_spectrum(predefined_frequency)
