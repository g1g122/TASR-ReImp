#!/bin/bash

# Architecture Experiments for Multi-Modal Emotion Recognition
# These experiments focus on architecture variations without projection

# Data path - change this to your dataset location
DATA_PATH="aligned_events.pkl"

# Base directory for checkpoints
BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

# Log file to track all runs
LOG_FILE="$BASE_DIR/architecture_experiments_log.txt"
echo "Starting architecture experiments at $(date)" > "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "Configuration Constants:" >> "$LOG_FILE"
echo "- Batch Size: 8" >> "$LOG_FILE"
echo "- Gradient Accumulation Steps: 8 (effective batch size: 64)" >> "$LOG_FILE"
echo "- Transformer Heads: 8" >> "$LOG_FILE"
echo "- Transformer Layers: 4" >> "$LOG_FILE"
echo "- No Projection Layers Used" >> "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"

# Function to run an experiment and log the results
run_experiment() {
    # Arguments:
    # $1: Experiment name
    # $2+: Command line arguments for pipeline.py
    
    exp_name=$1
    shift
    
    echo "Running experiment: $exp_name"
    echo "Command: python pipeline.py --data_path $DATA_PATH $@"
    echo "===========================================" 
    echo "Experiment: $exp_name" >> "$LOG_FILE"
    echo "Command: python pipeline.py --data_path $DATA_PATH $@" >> "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    
    # Run the experiment and capture output
    python pipeline.py --data_path "$DATA_PATH" "$@" | tee -a "$LOG_FILE"
    
    echo "Completed at: $(date)" >> "$LOG_FILE"
    echo "===========================================" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
}

# ============================================================
# SECTION A: Feed Forward Dimension Variations (no projection)
# ============================================================

# ============================================================
# Experiment A1: Binary with small FF dimension (1024)
# ============================================================
run_experiment "binary_small_ff" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 1024 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_small_ff"

# ============================================================
# Experiment A2: Binary with medium FF dimension (2048)
# ============================================================
run_experiment "binary_medium_ff" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_medium_ff"

# ============================================================
# Experiment A3: Binary with large FF dimension (3072)
# ============================================================
run_experiment "binary_large_ff" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 3072 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_large_ff"

# ============================================================
# Experiment A4: Valence-based with small FF dimension (1024)
# ============================================================
run_experiment "va5_small_ff" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 1024 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_small_ff"

# ============================================================
# Experiment A5: Valence-based with medium FF dimension (2048)
# ============================================================
run_experiment "va5_medium_ff" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_medium_ff"

# ============================================================
# Experiment A6: Valence-based with large FF dimension (3072)
# ============================================================
run_experiment "va5_large_ff" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 3072 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_large_ff"

# ============================================================
# Experiment A7: Arousal-based with small FF dimension (1024)
# ============================================================
run_experiment "ar5_small_ff" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 1024 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_small_ff"

# ============================================================
# Experiment A8: Arousal-based with medium FF dimension (2048)
# ============================================================
run_experiment "ar5_medium_ff" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_medium_ff"

# ============================================================
# Experiment A9: Arousal-based with large FF dimension (3072)
# ============================================================
run_experiment "ar5_large_ff" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 3072 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_large_ff"

# ============================================================
# SECTION B: Classifier Hidden Dimension Variations
# ============================================================

# ============================================================
# Experiment B1: Binary with small classifier dimension (256)
# ============================================================
run_experiment "binary_small_classifier" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 256 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_small_classifier"

# ============================================================
# Experiment B2: Binary with large classifier dimension (1024)
# ============================================================
run_experiment "binary_large_classifier" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 1024 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_large_classifier"

# ============================================================
# Experiment B3: Valence-based with small classifier dimension (256)
# ============================================================
run_experiment "va5_small_classifier" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 256 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_small_classifier"

# ============================================================
# Experiment B4: Valence-based with large classifier dimension (1024)
# ============================================================
run_experiment "va5_large_classifier" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 1024 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_large_classifier"

# ============================================================
# Experiment B5: Arousal-based with small classifier dimension (256)
# ============================================================
run_experiment "ar5_small_classifier" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 256 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_small_classifier"

# ============================================================
# Experiment B6: Arousal-based with large classifier dimension (1024)
# ============================================================
run_experiment "ar5_large_classifier" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 1024 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_large_classifier"

# ============================================================
# SECTION C: Data Split Variations
# ============================================================

# ============================================================
# Experiment C1: Binary with different data split (80/10/10)
# ============================================================
run_experiment "binary_split_80_10" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_split_80_10"

# ============================================================
# Experiment C2: Binary with different data split (60/20/20)
# ============================================================
run_experiment "binary_split_60_20" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --checkpoint_dir "$BASE_DIR/binary_split_60_20"

# ============================================================
# Experiment C3: Valence-based with different data split (80/10/10)
# ============================================================
run_experiment "va5_split_80_10" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_split_80_10"

# ============================================================
# Experiment C4: Valence-based with different data split (60/20/20)
# ============================================================
run_experiment "va5_split_60_20" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --checkpoint_dir "$BASE_DIR/va5_split_60_20"

# ============================================================
# Experiment C5: Arousal-based with different data split (80/10/10)
# ============================================================
run_experiment "ar5_split_80_10" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_split_80_10"

# ============================================================
# Experiment C6: Arousal-based with different data split (60/20/20)
# ============================================================
run_experiment "ar5_split_60_20" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.6 \
    --val_ratio 0.2 \
    --checkpoint_dir "$BASE_DIR/ar5_split_60_20"

echo "All architecture experiments completed!"
echo "Log file available at: $LOG_FILE" 