#!/bin/bash

# Projection Layer Experiments for Multi-Modal Emotion Recognition
# These experiments focus on variations with projection layers

# Data path - change this to your dataset location
DATA_PATH="aligned_events.pkl"

# Base directory for checkpoints
BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

# Log file to track all runs
LOG_FILE="$BASE_DIR/projection_experiments_log.txt"
echo "Starting projection experiments at $(date)" > "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "Configuration Constants:" >> "$LOG_FILE"
echo "- Batch Size: 8" >> "$LOG_FILE"
echo "- Gradient Accumulation Steps: 8 (effective batch size: 64)" >> "$LOG_FILE"
echo "- Transformer Heads: 8" >> "$LOG_FILE"
echo "- Transformer Layers: 4" >> "$LOG_FILE"
echo "- All experiments use projection layers" >> "$LOG_FILE"
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
# SECTION A: Projection Dimension Variations
# ============================================================

# ============================================================
# Experiment A1: Binary with small projection dimension (512)
# ============================================================
run_experiment "binary_small_proj" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 512 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_small_proj"

# ============================================================
# Experiment A2: Binary with medium projection dimension (1024)
# ============================================================
run_experiment "binary_medium_proj" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1024 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_medium_proj"

# ============================================================
# Experiment A3: Binary with large projection dimension (1536)
# ============================================================
run_experiment "binary_large_proj" \
    --task binary \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_large_proj"

# ============================================================
# Experiment A4: Valence-based with small projection dimension (512)
# ============================================================
run_experiment "va5_small_proj" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 512 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_small_proj"

# ============================================================
# Experiment A5: Valence-based with medium projection dimension (1024)
# ============================================================
run_experiment "va5_medium_proj" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1024 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_medium_proj"

# ============================================================
# Experiment A6: Valence-based with large projection dimension (1536)
# ============================================================
run_experiment "va5_large_proj" \
    --task va5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_large_proj"

# ============================================================
# Experiment A7: Arousal-based with small projection dimension (512)
# ============================================================
run_experiment "ar5_small_proj" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 512 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_small_proj"

# ============================================================
# Experiment A8: Arousal-based with medium projection dimension (1024)
# ============================================================
run_experiment "ar5_medium_proj" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1024 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_medium_proj"

# ============================================================
# Experiment A9: Arousal-based with large projection dimension (1536)
# ============================================================
run_experiment "ar5_large_proj" \
    --task ar5 \
    --feature_dim 512 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_large_proj"

# ============================================================
# SECTION B: Feature & Projection Combination Experiments
# ============================================================

# ============================================================
# Experiment B1: Binary with larger feature & projection dimensions
# ============================================================
run_experiment "binary_large_feat_proj" \
    --task binary \
    --feature_dim 768 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_large_feat_proj"

# ============================================================
# Experiment B2: Valence-based with larger feature & projection dimensions
# ============================================================
run_experiment "va5_large_feat_proj" \
    --task va5 \
    --feature_dim 768 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_large_feat_proj"

# ============================================================
# Experiment B3: Arousal-based with larger feature & projection dimensions
# ============================================================
run_experiment "ar5_large_feat_proj" \
    --task ar5 \
    --feature_dim 768 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2048 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 512 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_large_feat_proj"

# ============================================================
# Experiment B4: Binary with larger feature, proj & FF dimensions
# ============================================================
run_experiment "binary_large_all" \
    --task binary \
    --feature_dim 768 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 3072 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 768 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_large_all"

# ============================================================
# Experiment B5: Valence-based with larger feature, proj & FF dimensions
# ============================================================
run_experiment "va5_large_all" \
    --task va5 \
    --feature_dim 768 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 3072 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 768 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_large_all"

# ============================================================
# Experiment B6: Arousal-based with larger feature, proj & FF dimensions
# ============================================================
run_experiment "ar5_large_all" \
    --task ar5 \
    --feature_dim 768 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 3072 \
    --use_projection \
    --projection_dim 1536 \
    --classifier_hidden_dim 768 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_large_all"

echo "All projection experiments completed!"
echo "Log file available at: $LOG_FILE" 