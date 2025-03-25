#!/bin/bash

# Multi-Modal Emotion Recognition Experiment Runner
# This script runs multiple experiments with different hyperparameter configurations

# Data path - change this to your dataset location
DATA_PATH="aligned_events.pkl"

# Base directory for checkpoints
BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

# Log file to track all runs
LOG_FILE="$BASE_DIR/all_experiments_log.txt"
echo "Starting experiments at $(date)" > "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "Classification Tasks:" >> "$LOG_FILE"
echo "1. Binary Classification (PANAS-based): Classes determined by comparing sums of positive affect PANAS items (indices 4, 6, 7, 9) vs negative affect PANAS items (indices 0, 1, 2, 3, 5, 8)" >> "$LOG_FILE"
echo "2. 5-Class Classification (Valence-based): Classes directly mapped from valence scores (1-5) to class indices (0-4)" >> "$LOG_FILE"
echo "3. 5-Class Classification (Arousal-based): Classes directly mapped from arousal scores (1-5) to class indices (0-4)" >> "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "Data Split: 70% Training, 10% Validation, 20% Testing" >> "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"

# Function to run an experiment and log the results
run_experiment() {
    # Arguments:
    # $1: Experiment name
    # $2+: Command line arguments for pipeline.py
    
    exp_name=$1
    shift
    
    echo "Running experiment: $exp_name"
    echo "Command: python pipeline.py --data_path $DATA_PATH --train_ratio 0.7 --val_ratio 0.1 $@"
    echo "===========================================" 
    echo "Experiment: $exp_name" >> "$LOG_FILE"
    echo "Command: python pipeline.py --data_path $DATA_PATH --train_ratio 0.7 --val_ratio 0.1 $@" >> "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    
    # Run the experiment and capture output
    python pipeline.py --data_path "$DATA_PATH" --train_ratio 0.7 --val_ratio 0.1 "$@" | tee -a "$LOG_FILE"
    
    echo "Completed at: $(date)" >> "$LOG_FILE"
    echo "===========================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

# ============================================================
# SECTION A: Default Experiments
# ============================================================

# ============================================================
# Experiment A1: Binary classification (default hyperparameters)
# ============================================================
run_experiment "binary_default" \
    --task binary \
    --checkpoint_dir "$BASE_DIR/binary_default"

# ============================================================
# Experiment A2: 5-class Valence-based classification (default hyperparameters)
# ============================================================
run_experiment "va5_default" \
    --task va5 \
    --checkpoint_dir "$BASE_DIR/va5_default"

# ============================================================
# Experiment A3: 5-class Arousal-based classification (default hyperparameters)
# ============================================================
run_experiment "ar5_default" \
    --task ar5 \
    --checkpoint_dir "$BASE_DIR/ar5_default"

# ============================================================
# SECTION B: Model Architecture Experiments
# ============================================================

# ============================================================
# Experiment B1: Binary classification with larger model
# ============================================================
run_experiment "binary_large" \
    --task binary \
    --feature_dim 768 \
    --num_heads 12 \
    --num_layers 6 \
    --d_ff 3072 \
    --classifier_hidden_dim 768 \
    --checkpoint_dir "$BASE_DIR/binary_large"

# ============================================================
# Experiment B2: Valence-based classification with larger model
# ============================================================
run_experiment "va5_large" \
    --task va5 \
    --feature_dim 768 \
    --num_heads 12 \
    --num_layers 6 \
    --d_ff 3072 \
    --classifier_hidden_dim 768 \
    --checkpoint_dir "$BASE_DIR/va5_large"

# ============================================================
# Experiment B3: Arousal-based classification with larger model
# ============================================================
run_experiment "ar5_large" \
    --task ar5 \
    --feature_dim 768 \
    --num_heads 12 \
    --num_layers 6 \
    --d_ff 3072 \
    --classifier_hidden_dim 768 \
    --checkpoint_dir "$BASE_DIR/ar5_large"

# ============================================================
# Experiment B4: Binary classification with projection layer
# ============================================================
run_experiment "binary_projection" \
    --task binary \
    --use_projection \
    --projection_dim 1024 \
    --checkpoint_dir "$BASE_DIR/binary_projection"

# ============================================================
# Experiment B5: Valence-based classification with projection layer
# ============================================================
run_experiment "va5_projection" \
    --task va5 \
    --use_projection \
    --projection_dim 1024 \
    --checkpoint_dir "$BASE_DIR/va5_projection"

# ============================================================
# Experiment B6: Arousal-based classification with projection layer
# ============================================================
run_experiment "ar5_projection" \
    --task ar5 \
    --use_projection \
    --projection_dim 1024 \
    --checkpoint_dir "$BASE_DIR/ar5_projection"

# ============================================================
# Experiment B7: Binary classification with smaller model
# ============================================================
run_experiment "binary_small" \
    --task binary \
    --feature_dim 256 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 1024 \
    --classifier_hidden_dim 256 \
    --checkpoint_dir "$BASE_DIR/binary_small"

# ============================================================
# Experiment B8: Valence-based classification with smaller model
# ============================================================
run_experiment "va5_small" \
    --task va5 \
    --feature_dim 256 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 1024 \
    --classifier_hidden_dim 256 \
    --checkpoint_dir "$BASE_DIR/va5_small"

# ============================================================
# Experiment B9: Arousal-based classification with smaller model
# ============================================================
run_experiment "ar5_small" \
    --task ar5 \
    --feature_dim 256 \
    --num_heads 4 \
    --num_layers 2 \
    --d_ff 1024 \
    --classifier_hidden_dim 256 \
    --checkpoint_dir "$BASE_DIR/ar5_small"

# ============================================================
# SECTION C: Optimization Experiments
# ============================================================

# ============================================================
# Experiment C1: Binary classification with lower learning rate
# ============================================================
run_experiment "binary_low_lr" \
    --task binary \
    --learning_rate 5e-4 \
    --checkpoint_dir "$BASE_DIR/binary_low_lr"

# ============================================================
# Experiment C2: Valence-based classification with lower learning rate
# ============================================================
run_experiment "va5_low_lr" \
    --task va5 \
    --learning_rate 5e-4 \
    --checkpoint_dir "$BASE_DIR/va5_low_lr"

# ============================================================
# Experiment C3: Arousal-based classification with lower learning rate
# ============================================================
run_experiment "ar5_low_lr" \
    --task ar5 \
    --learning_rate 5e-4 \
    --checkpoint_dir "$BASE_DIR/ar5_low_lr"

# ============================================================
# Experiment C4: Binary classification with higher learning rate
# ============================================================
run_experiment "binary_high_lr" \
    --task binary \
    --learning_rate 2e-3 \
    --checkpoint_dir "$BASE_DIR/binary_high_lr"

# ============================================================
# Experiment C5: Valence-based classification with higher learning rate
# ============================================================
run_experiment "va5_high_lr" \
    --task va5 \
    --learning_rate 2e-3 \
    --checkpoint_dir "$BASE_DIR/va5_high_lr"

# ============================================================
# Experiment C6: Arousal-based classification with higher learning rate
# ============================================================
run_experiment "ar5_high_lr" \
    --task ar5 \
    --learning_rate 2e-3 \
    --checkpoint_dir "$BASE_DIR/ar5_high_lr"

# ============================================================
# SECTION D: Regularization Experiments
# ============================================================

# ============================================================
# Experiment D1: Binary classification with higher dropout
# ============================================================
run_experiment "binary_high_dropout" \
    --task binary \
    --dropout 0.3 \
    --weight_decay 5e-5 \
    --checkpoint_dir "$BASE_DIR/binary_high_dropout"

# ============================================================
# Experiment D2: Valence-based classification with higher dropout
# ============================================================
run_experiment "va5_high_dropout" \
    --task va5 \
    --dropout 0.3 \
    --weight_decay 5e-5 \
    --checkpoint_dir "$BASE_DIR/va5_high_dropout"

# ============================================================
# Experiment D3: Arousal-based classification with higher dropout
# ============================================================
run_experiment "ar5_high_dropout" \
    --task ar5 \
    --dropout 0.3 \
    --weight_decay 5e-5 \
    --checkpoint_dir "$BASE_DIR/ar5_high_dropout"

# ============================================================
# SECTION E: Batch Size Experiments
# ============================================================

# ============================================================
# Experiment E1: Binary classification with smaller batch size
# ============================================================
run_experiment "binary_small_batch" \
    --task binary \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --checkpoint_dir "$BASE_DIR/binary_small_batch"

# ============================================================
# Experiment E2: Valence-based classification with smaller batch size
# ============================================================
run_experiment "va5_small_batch" \
    --task va5 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --checkpoint_dir "$BASE_DIR/va5_small_batch"

# ============================================================
# Experiment E3: Arousal-based classification with smaller batch size
# ============================================================
run_experiment "ar5_small_batch" \
    --task ar5 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --checkpoint_dir "$BASE_DIR/ar5_small_batch"

# ============================================================
# Experiment E4: Binary classification with larger batch size
# ============================================================
run_experiment "binary_large_batch" \
    --task binary \
    --batch_size 128 \
    --learning_rate 2e-3 \
    --checkpoint_dir "$BASE_DIR/binary_large_batch"

# ============================================================
# Experiment E5: Valence-based classification with larger batch size
# ============================================================
run_experiment "va5_large_batch" \
    --task va5 \
    --batch_size 128 \
    --learning_rate 2e-3 \
    --checkpoint_dir "$BASE_DIR/va5_large_batch"

# ============================================================
# Experiment E6: Arousal-based classification with larger batch size
# ============================================================
run_experiment "ar5_large_batch" \
    --task ar5 \
    --batch_size 128 \
    --learning_rate 2e-3 \
    --checkpoint_dir "$BASE_DIR/ar5_large_batch"

# ============================================================
# SECTION F: Combined Parameter Experiments
# ============================================================

# ============================================================
# Experiment F1: Binary classification - best estimated combination
# ============================================================
run_experiment "binary_best_estimate" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1024 \
    --learning_rate 1e-3 \
    --dropout 0.2 \
    --batch_size 64 \
    --checkpoint_dir "$BASE_DIR/binary_best_estimate"

# ============================================================
# Experiment F2: Valence-based classification - best estimated combination
# ============================================================
run_experiment "va5_best_estimate" \
    --task va5 \
    --feature_dim 768 \
    --num_heads 12 \
    --num_layers 6 \
    --d_ff 3072 \
    --use_projection \
    --projection_dim 1536 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 64 \
    --checkpoint_dir "$BASE_DIR/va5_best_estimate"

# ============================================================
# Experiment F3: Arousal-based classification - best estimated combination
# ============================================================
run_experiment "ar5_best_estimate" \
    --task ar5 \
    --feature_dim 768 \
    --num_heads 12 \
    --num_layers 6 \
    --d_ff 3072 \
    --use_projection \
    --projection_dim 1536 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 64 \
    --checkpoint_dir "$BASE_DIR/ar5_best_estimate"

echo "All experiments completed!"
echo "Log file available at: $LOG_FILE"

# Create a summary report of all experiment results
echo "Generating summary report..."
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
        
        # Create entry
        entry = {
            'Experiment': exp_dir,
            'Task': config['task'],
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'F1 Score': test_metrics['f1_score'],
            'Loss': test_metrics['loss'],
            'Feature Dim': config['feature_dim'],
            'Num Heads': config['num_heads'],
            'Num Layers': config['num_layers'],
            'Learning Rate': config['learning_rate'],
            'Batch Size': config['batch_size'],
            'Dropout': config['dropout'],
            'Use Projection': config['use_projection'],
            'Parameters': 'N/A'  # Will be filled in later
        }
        
        results.append(entry)
    except Exception as e:
        print(f'Error processing {exp_dir}: {e}')

# Create DataFrame and save to CSV
if results:
    df = pd.DataFrame(results)
    
    # Add a column classifying experiments by parameter category
    def categorize_experiment(exp_name):
        if 'large' in exp_name and 'batch' not in exp_name:
            return 'Large Model'
        elif 'small' in exp_name and 'batch' not in exp_name:
            return 'Small Model'
        elif 'projection' in exp_name:
            return 'With Projection'
        elif 'dropout' in exp_name:
            return 'Dropout Variation'
        elif 'lr' in exp_name:
            return 'Learning Rate Variation'
        elif 'batch' in exp_name:
            return 'Batch Size Variation'
        elif 'best' in exp_name:
            return 'Combined Parameters'
        else:
            return 'Default'
    
    df['Category'] = df['Experiment'].apply(categorize_experiment)
    
    # Sort by task and F1 score
    df = df.sort_values(['Task', 'F1 Score'], ascending=[True, False])
    
    # Add an experiment ID for easier reference
    df['ID'] = range(1, len(df) + 1)
    
    # Reorder columns
    df = df[['ID', 'Experiment', 'Category', 'Task', 'Accuracy', 'Precision', 'F1 Score', 
             'Loss', 'Feature Dim', 'Num Heads', 'Num Layers', 'Learning Rate', 
             'Batch Size', 'Dropout', 'Use Projection']]
    
    # Save complete results
    csv_file = os.path.join(base_dir, 'experiment_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f'Summary saved to {csv_file}')
    
    # Display best results for each category
    print('\nBest results by category:')
    
    task_categories = df.groupby(['Task', 'Category'])
    
    for (task, category), group in task_categories:
        best_in_category = group.iloc[0]  # Get the best result (already sorted by F1 score)
        print(f'Task: {task}, Category: {category}')
        print(f'  Experiment: {best_in_category[\"Experiment\"]}')
        print(f'  F1 Score: {best_in_category[\"F1 Score\"]:.4f}')
        print(f'  Accuracy: {best_in_category[\"Accuracy\"]:.4f}')
    
    # Display overall best results
    print('\nBest overall results for binary classification:')
    binary_df = df[df['Task'] == 'binary'].head(3)
    print(binary_df[['ID', 'Experiment', 'Category', 'Accuracy', 'F1 Score']].to_string(index=False))
    
    print('\nBest overall results for 5-class classification:')
    va5_df = df[df['Task'] == 'va5'].head(3)
    print(va5_df[['ID', 'Experiment', 'Category', 'Accuracy', 'F1 Score']].to_string(index=False))
    
    # Generate performance by category summary
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
else:
    print('No results found!')
" 