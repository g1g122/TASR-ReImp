"""
Author: Etlazure
Creation Date: February 24, 2025
Purpose: Align denoised physiological data with ESM events, extracting the 30 minutes before each ESM event
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define the data directories
DENOISED_DATA_DIR = './denoised_data'
ESM_INFO_PATH = '../raw_dataset/DAPPER/Psychol_Rec/ESM.xlsx'
OUTPUT_FILE = 'aligned_events.pkl'

# Read ESM data
print("Reading ESM data...")
ESM_info = pd.read_excel(ESM_INFO_PATH)

# Get participant IDs from the denoised data directory
ids = [id_dir for id_dir in os.listdir(DENOISED_DATA_DIR) if os.path.isdir(os.path.join(DENOISED_DATA_DIR, id_dir))]

aligned_events = {}
total_events = 0
missing_data = 0

print(f"Processing {len(ids)} participants...")
for id_str in ids:
    print(f"Processing participant {id_str}")
    ind_denoised = os.path.join(DENOISED_DATA_DIR, id_str)

    # Read and combine all CSV files for this participant
    dfs = []
    for csv_file in os.listdir(ind_denoised):
        if len(csv_file) == 33:  # This checks if it's a valid data file based on filename length
            csv_file_path = os.path.join(ind_denoised, csv_file)
            df = pd.read_csv(csv_file_path, header=0, sep=',')
            df['time'] = pd.to_datetime(df['time'])  # Convert time column to datetime
            dfs.append(df)
    
    if not dfs:
        print(f"No data files found for participant {id_str}, skipping")
        continue
        
    # Merge all dataframes and sort by time
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.sort_values('time')
    merged_df = merged_df.reset_index(drop=True)
    merged_df.set_index('time', inplace=True)  # Set time as index for easier slicing

    # Get ESM records for this participant
    id = int(id_str)
    ind_esm = ESM_info[ESM_info['Participant ID'] == id]
    
    if ind_esm.empty:
        print(f"No ESM records found for participant {id}, skipping")
        continue

    id_events = {}
    for event, (idx, row) in enumerate(ind_esm.iterrows(), 1):
        # Extract PANAS, Valence, and Arousal scores
        panas_cols = [f'PANAS_{i}' for i in range(1, 11)]
        panas_scores = row[panas_cols].values
        valence = row['Valence']
        arousal = row['Arousal']

        # Parse the end time (which is the ESM start time)
        end_time_str = row[' StartTime '].strip()
        end_time = datetime.strptime(end_time_str, '%Y/%m/%d %H:%M:%S')
        start_time = end_time - timedelta(minutes=30)

        # Create a complete time range with 1-second intervals
        full_time_range = pd.date_range(start=start_time + timedelta(seconds=1), end=end_time, freq='s')
        
        # Get data for this time range
        try:
            time_slice = merged_df.loc[start_time + timedelta(seconds=1):end_time, ['heart_rate', 'motion', 'GSR']]
        except KeyError:
            print(f"Warning: Missing data columns for ID {id} event {event}")
            continue

        # Check if we have data for this time range
        if len(time_slice) == 0:
            print(f"Warning: No data found for ID {id} event {event} between {start_time} and {end_time}")
            missing_data += 1
            continue
            
        # Reindex to show missing time points
        time_slice = time_slice.reindex(full_time_range)
        
        # Fill missing values
        time_slice['heart_rate'] = time_slice['heart_rate'].ffill().bfill()
        time_slice['GSR'] = time_slice['GSR'].ffill().bfill()
        time_slice['motion'] = time_slice['motion'].ffill().bfill()  # Forward then backward fill

        # Check if we have enough data after filling
        if len(time_slice) < 1800:  # 30 minutes * 60 seconds
            print(f"Warning: Insufficient data points for ID {id} event {event} after filling")
            continue

        # Convert to numpy arrays
        heart_rate = np.array(time_slice['heart_rate'])
        motion = np.array(time_slice['motion'])
        GSR = np.array(time_slice['GSR'])

        # Store the data for this event
        id_events[f'event{event}'] = {
            'heart_rate': heart_rate,
            'motion': motion,
            'GSR': GSR,
            'panas': panas_scores,
            'valence': valence,
            'arousal': arousal,
            'end_time': end_time_str
        }

    if id_events:  # Only add if there are events for this participant
        aligned_events[id] = id_events
        total_events += len(id_events)

# Save the aligned data to a pickle file
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(aligned_events, f)

# Output statistics
print(f"\nProcessing complete!")
print(f"Number of participants with valid data: {len(aligned_events)}")
print(f"Total number of aligned events: {total_events}")
print(f"Number of events with missing data: {missing_data}")

# Display data structure for verification
print("\nData structure for first participant:")
if aligned_events:
    first_id = next(iter(aligned_events))
    first_event = next(iter(aligned_events[first_id]))
    print(f"Participant ID: {first_id}")
    print(f"Event: {first_event}")
    for key, value in aligned_events[first_id][first_event].items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: np.ndarray {value.shape}")
        else:
            print(f"  {key}: {value}") 