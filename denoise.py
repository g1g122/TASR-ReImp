"""
Author: Etlazure
Creation Date: February 16, 2025
Purpose: Denoise physiological data from the DAPPER dataset using adaptive noise cancellation (LMS algorithm)
         and moving median filter
"""

import os
import pandas as pd
import numpy as np
import padasip as pa
from padasip.filters import FilterNLMS
from scipy.signal import medfilt
from pathlib import Path
from tqdm import tqdm

# Define the main data directory
RAW_DATA_DIR = '../raw_dataset/DAPPER/Physiol_Rec'
OUTPUT_DIR = './denoised_data'

def apply_lms_filter(signal, n=10, mu=0.1):
    """Apply adaptive LMS filter to signal"""
    if len(signal) < n:
        return signal  # Avoid processing signals that are too short
    
    # Normalize input signal
    # signal_mean = np.mean(signal)
    # signal_std = np.std(signal)
    # normalized_signal = (signal - signal_mean) / (signal_std + 1e-8)  # Avoid division by zero
    
    # Generate input matrix, each sample is a window of past n points
    x = pa.preprocess.input_from_history(signal, n)
    # Expected output is the original signal with the first n-1 samples removed
    d = signal[n-1:]
    
    # Create filter
    f = FilterNLMS(n=n, mu=mu, w="zeros")
    
    # Train filter
    try:
        y, e, w = f.run(d, x)
    except Exception as e:
        print(f"Error in LMS filter: {e}")
        return signal  # Return original signal when error occurs
    
    # Align denoised result with original signal length
    denoised_signal = np.zeros_like(signal)
    denoised_signal[:n-1] = signal[:n-1]  # Keep the first n-1 samples unchanged
    denoised_signal[n-1:] = y             # The rest are filtered results
    # denoised_signal[n-1:] = y * signal_std + signal_mean  # Restore original scale
    return denoised_signal

def apply_median_filter(signal, kernel_size=3):
    """Apply moving median filter to signal"""
    return medfilt(signal, kernel_size)

def denoise_file(file_path, output_path):
    """Denoise a single CSV file and save the results"""
    # Read the CSV file
    df = pd.read_csv(file_path, header=0, sep=',')
    print(df.head())
    
    # Convert time column to datetime type and handle errors
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Handle NaT values similar to the preprocess.py script
    mask = df['time'].isna()
    df.loc[mask, 'time'] = df['time'].shift(-1) - pd.Timedelta(seconds=1)
    
    # Sort by time
    df = df.sort_values('time')
    df = df.reset_index(drop=True)
    
    # Make a copy for the denoised data
    denoised_df = df.copy()
    
    # Apply LMS and median filter to heart_rate
    if len(df['heart_rate']) > 0:
        print("Applying LMS filter to heart_rate")
        # 调整LMS参数：大幅增加n以捕获更长期的趋势，显著减小mu以获得平滑效果
        denoised_heart_rate = apply_lms_filter(df['heart_rate'].values, n=128, mu=0.0001)
        # 保持中值滤波kernel_size=3
        denoised_heart_rate = apply_median_filter(denoised_heart_rate)
        denoised_df['heart_rate'] = denoised_heart_rate
    
    # Apply LMS and median filter to motion
    if len(df['motion']) > 0:
        print("Applying LMS filter to motion")
        denoised_motion = apply_lms_filter(df['motion'].values, n=8, mu=0.02)
        denoised_motion = apply_median_filter(denoised_motion)
        denoised_df['motion'] = denoised_motion
    
    # Apply LMS and median filter to GSR
    if len(df['GSR']) > 0:
        print("Applying LMS filter to GSR")
        denoised_gsr = apply_lms_filter(df['GSR'].values, n=32, mu=0.01)
        denoised_gsr = apply_median_filter(denoised_gsr)
        denoised_df['GSR'] = denoised_gsr

    denoised_df.drop('battery_info', axis=1, inplace=True)
    
    # Save the denoised data
    denoised_df.to_csv(output_path, index=False)

def main():
    """Main function to process all files"""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of participant IDs
    ids = [id_dir for id_dir in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, id_dir))]
    
    for id_str in tqdm(ids, desc="Processing participants"):
        # Skip participant 3029 as done in preprocess.py
        if id_str == '3029':
            continue
        print(f"Processing participant {id_str}")
        
        # Create output directory for this participant
        participant_output_dir = os.path.join(OUTPUT_DIR, id_str)
        os.makedirs(participant_output_dir, exist_ok=True)
        
        # Path to the participant's raw data
        participant_raw_dir = os.path.join(RAW_DATA_DIR, id_str)
        
        # Process each CSV file for this participant
        for csv_file in os.listdir(participant_raw_dir):
            if len(csv_file) == 33:
                input_path = os.path.join(participant_raw_dir, csv_file)
                output_path = os.path.join(participant_output_dir, csv_file)
                
                try:
                    denoise_file(input_path, output_path)
                except Exception as e:
                    print(f"Error processing file {input_path}: {e}")

if __name__ == "__main__":
    main()
    print("Denoising complete!") 