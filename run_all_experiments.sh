#!/bin/bash

# Master script to run all experiments for multi-modal emotion recognition

echo "============================================================"
echo "Starting all experiments for multi-modal emotion recognition"
echo "============================================================"
echo "GPU memory constraints: batch_size=8, num_heads=8, num_layers=4"
echo "Using gradient accumulation (steps=8) for effective batch size of 64"
echo "============================================================"

# Create timestamp folder for all results
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "experiment_results"

# Set BASE_DIR environment variable for all scripts
export BASE_DIR="experiment_results"
echo "All results will be saved to: $BASE_DIR"
echo "============================================================"

# Run all experiment scripts in sequence
echo "1. Running architecture experiments (without projection)..."
bash run_architecture_experiments.sh
echo "Architecture experiments completed!"
echo "============================================================"

echo "2. Running projection layer experiments..."
bash run_projection_experiments.sh
echo "Projection layer experiments completed!"
echo "============================================================"

echo "3. Running training settings experiments..."
bash run_training_experiments.sh
echo "Training settings experiments completed!"
echo "============================================================"

echo "All experiments completed successfully!"
echo "Results are in: $BASE_DIR"
echo "============================================================"

# Generate final summary report
echo "Generating final summary report..."
python -c "
import os
import pickle
import pandas as pd
import numpy as np

# Base directory
base_dir = '$BASE_DIR'

# Collect results
results = []

for exp_dir in os.listdir(base_dir):
    full_path = os.path.join(base_dir, exp_dir)
    if not os.path.isdir(full_path):
        continue
    
    # Look for results pickle file
    results_file = os.path.join(full_path, 'results.pkl')
    if not os.path.exists(results_file):
        continue
    
    try:
        # Load the results
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        # Get test metrics
        test_metrics = data['test_metrics']
        config = data['config']
        
        # Calculate total parameters
        parameters = 'N/A'
        if 'model_parameters' in data:
            parameters = data['model_parameters']
            
        # Get effective batch size
        effective_batch_size = config['batch_size']
        if 'accumulation_steps' in config:
            effective_batch_size *= config['accumulation_steps']
        
        # Create entry
        entry = {
            'Experiment': exp_dir,
            'Task': config['task'],
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'F1 Score': test_metrics['f1_score'],
            'Loss': test_metrics['loss'],
            'Feature Dim': config['feature_dim'],
            'Use Projection': config['use_projection'],
            'Projection Dim': config.get('projection_dim', 'N/A'),
            'Num Heads': config['num_heads'],
            'Num Layers': config['num_layers'],
            'Feed-Forward Dim': config['d_ff'],
            'Classifier Hidden Dim': config['classifier_hidden_dim'],
            'Learning Rate': config['learning_rate'],
            'Dropout': config['dropout'],
            'Weight Decay': config['weight_decay'],
            'Batch Size': config['batch_size'],
            'Effective Batch Size': effective_batch_size,
            'Train Ratio': config['train_ratio'],
            'Val Ratio': config['val_ratio'],
            'Parameters': parameters
        }
        
        results.append(entry)
    except Exception as e:
        print(f'Error processing {exp_dir}: {e}')

# Create DataFrame and save to CSV
if results:
    df = pd.DataFrame(results)
    
    # Add a column classifying experiments by parameter category
    def categorize_experiment(exp_name):
        if 'ff' in exp_name:
            return 'Feed-Forward Dimension'
        elif 'classifier' in exp_name:
            return 'Classifier Dimension'
        elif 'split' in exp_name:
            return 'Data Split'
        elif 'proj' in exp_name:
            return 'Projection Dimension'
        elif 'feat' in exp_name and 'proj' in exp_name:
            return 'Feature & Projection'
        elif 'large_all' in exp_name:
            return 'All Large Dimensions'
        elif 'lr' in exp_name:
            return 'Learning Rate'
        elif 'dropout' in exp_name:
            return 'Dropout Rate'
        elif 'wd' in exp_name:
            return 'Weight Decay'
        elif 'accum' in exp_name:
            return 'Gradient Accumulation'
        else:
            return 'Other'
    
    df['Category'] = df['Experiment'].apply(categorize_experiment)
    
    # Sort by task and F1 score
    df = df.sort_values(['Task', 'F1 Score'], ascending=[True, False])
    
    # Add an experiment ID for easier reference
    df['ID'] = range(1, len(df) + 1)
    
    # Reorder columns
    column_order = ['ID', 'Experiment', 'Category', 'Task', 'Accuracy', 'Precision', 
                   'F1 Score', 'Loss', 'Feature Dim', 'Use Projection', 'Projection Dim',
                   'Num Heads', 'Num Layers', 'Feed-Forward Dim', 'Classifier Hidden Dim',
                   'Learning Rate', 'Dropout', 'Weight Decay', 'Batch Size', 
                   'Effective Batch Size', 'Train Ratio', 'Val Ratio', 'Parameters']
    df = df[[col for col in column_order if col in df.columns]]
    
    # Save complete results
    csv_file = os.path.join(base_dir, 'all_experiment_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f'Summary saved to {csv_file}')
    
    # Display best results for each task
    print('\nBest results by task:')
    
    for task in df['Task'].unique():
        task_df = df[df['Task'] == task].head(3)
        print(f'\nBest results for {task} classification:')
        print(task_df[['ID', 'Experiment', 'Category', 'Accuracy', 'F1 Score']].to_string(index=False))
    
    # Display best results for each category within task
    print('\nBest results by task and category:')
    
    task_categories = df.groupby(['Task', 'Category'])
    
    for (task, category), group in task_categories:
        best_in_category = group.iloc[0]  # Get the best result (already sorted by F1 score)
        print(f'Task: {task}, Category: {category}')
        print(f'  Experiment: {best_in_category[\"Experiment\"]}')
        print(f'  F1 Score: {best_in_category[\"F1 Score\"]:.4f}')
        print(f'  Accuracy: {best_in_category[\"Accuracy\"]:.4f}')
    
    # Generate category performance summary
    print('\nGenerating category performance summary...')
    category_summary = df.groupby(['Task', 'Category']).agg({
        'Accuracy': ['mean', 'max'],
        'F1 Score': ['mean', 'max'],
        'ID': 'count'
    }).reset_index()
    
    category_summary.columns = ['Task', 'Category', 'Mean Accuracy', 'Max Accuracy', 
                               'Mean F1 Score', 'Max F1 Score', 'Count']
    
    category_file = os.path.join(base_dir, 'category_summary.csv')
    category_summary.to_csv(category_file, index=False)
    print(f'Category summary saved to {category_file}')
    
    # Create recommendation for best parameters
    print('\nRecommended best parameters:')
    
    for task in df['Task'].unique():
        best = df[df['Task'] == task].iloc[0]
        print(f'\nFor {task} classification:')
        print(f'  Model: {best[\"Feature Dim\"]} feature dim, {best[\"Projection Dim\"]} projection dim')
        print(f'  Architecture: {best[\"Num Heads\"]} heads, {best[\"Num Layers\"]} layers, {best[\"Feed-Forward Dim\"]} FF dim')
        print(f'  Training: LR={best[\"Learning Rate\"]}, dropout={best[\"Dropout\"]}, weight decay={best[\"Weight Decay\"]}')
        print(f'  From experiment: {best[\"Experiment\"]} (F1={best[\"F1 Score\"]:.4f}, Acc={best[\"Accuracy\"]:.4f})')
else:
    print('No results found!')
" 

echo "All done!" 