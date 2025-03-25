"""
Author: Etlazure
Creation Date: February 18, 2025
Purpose: Visualize denoised physiological data for a specific participant for each day
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, time
import glob
from pathlib import Path

# Define directories
DENOISED_DATA_DIR = './denoised_data'

# Parameters
PARTICIPANT_ID = '1004'
START_TIME = time(17, 0)  # 17:00
END_TIME = time(20, 0)    # 20:00

def load_and_filter_data(directory, participant_id):
    """
    Load data from CSV files
    """
    # Path to participant's data
    participant_dir = os.path.join(directory, participant_id)
    
    # Get all CSV files for this participant
    csv_files = glob.glob(os.path.join(participant_dir, "*.csv"))
    
    # Read and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        if len(os.path.basename(csv_file)) == 33:
            df = pd.read_csv(csv_file, header=0, sep=',')
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            dfs.append(df)
    
    # Concatenate all dataframes
    if not dfs:
        raise ValueError(f"No data found for participant {participant_id}")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by time and handle NaT values
    mask = combined_df['time'].isna()
    combined_df.loc[mask, 'time'] = combined_df['time'].shift(-1) - pd.Timedelta(seconds=1)
    combined_df = combined_df.sort_values('time')
    combined_df = combined_df.reset_index(drop=True)
    
    return combined_df

def get_daily_data(df, start_time, end_time):
    """
    Split the data by day and filter by time range for each day
    """
    # Add date column
    df['date'] = df['time'].dt.date
    
    # Get unique dates
    unique_dates = df['date'].unique()
    
    # Dictionary to store daily data
    daily_data = {}
    
    for date in unique_dates:
        # Filter data for this date and time range
        daily_df = df[
            (df['date'] == date) & 
            (df['time'].dt.time >= start_time) & 
            (df['time'].dt.time <= end_time)
        ].copy()
        
        # Only include days with sufficient data
        if len(daily_df) > 10:  # Arbitrary threshold to ensure meaningful visualization
            daily_data[date] = daily_df
    
    return daily_data

def plot_signal(df, signal_name, date_str):
    """
    Create plot for a specific denoised signal for a specific date
    """
    plt.figure(figsize=(15, 6))
    
    # Plot denoised signal
    plt.plot(df['time'], df[signal_name], 'r-', label='Denoised')
    
    # Format x-axis to show time
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    
    # Add labels and legend
    plt.title(f'Denoised {signal_name} for Participant {PARTICIPANT_ID} - {date_str}')
    plt.xlabel('Time')
    plt.ylabel(signal_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'participant_{PARTICIPANT_ID}_{signal_name}_denoised_{date_str}.png'), dpi=300)
    plt.close()

def plot_combined_signals(df, date_str):
    """
    Create a combined figure with all three signals for a specific date
    """
    plt.figure(figsize=(15, 12))
    
    # Heart rate subplot
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['heart_rate'], 'r-')
    plt.title(f'Denoised Heart Rate - Participant {PARTICIPANT_ID} - {date_str}')
    plt.ylabel('Heart Rate')
    plt.grid(True, alpha=0.3)
    
    # Motion subplot
    plt.subplot(3, 1, 2)
    plt.plot(df['time'], df['motion'], 'r-')
    plt.title(f'Denoised Motion - Participant {PARTICIPANT_ID} - {date_str}')
    plt.ylabel('Motion')
    plt.grid(True, alpha=0.3)
    
    # GSR subplot
    plt.subplot(3, 1, 3)
    plt.plot(df['time'], df['GSR'], 'r-')
    plt.title(f'Denoised GSR - Participant {PARTICIPANT_ID} - {date_str}')
    plt.xlabel('Time')
    plt.ylabel('GSR')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show time on all subplots
    for i in range(1, 4):
        plt.subplot(3, 1, i)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    
    # Save the combined figure
    plt.tight_layout()
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'participant_{PARTICIPANT_ID}_all_signals_denoised_{date_str}.png'), dpi=300)
    plt.close()

def main():
    # Create output directory
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load all denoised data
        print(f"Loading denoised data for participant {PARTICIPANT_ID}...")
        denoised_df = load_and_filter_data(DENOISED_DATA_DIR, PARTICIPANT_ID)
        
        # Split data by day and filter by time range
        daily_data = get_daily_data(denoised_df, START_TIME, END_TIME)
        
        print(f"Found data for {len(daily_data)} days in the specified time range")
        
        # Process each day's data
        for date, df in daily_data.items():
            date_str = date.strftime('%Y-%m-%d')
            print(f"Processing data for {date_str}...")
            
            # Create individual plots for each signal
            for signal in ['heart_rate', 'motion', 'GSR']:
                if signal in df.columns:
                    print(f"  Creating plot for denoised {signal}...")
                    plot_signal(df, signal, date_str)
                else:
                    print(f"  Signal {signal} not found in dataset")
            
            # Create combined plot for all signals
            print(f"  Creating combined plot for all signals...")
            plot_combined_signals(df, date_str)
        
        print(f"Visualization completed. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    main() 