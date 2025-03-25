#!/bin/bash

# Training Settings Experiments for Multi-Modal Emotion Recognition
# These experiments focus on variations in training parameters

# Data path - change this to your dataset location
DATA_PATH="aligned_events.pkl"

# Base directory for checkpoints
BASE_DIR="experiment_results"
mkdir -p "$BASE_DIR"

# Log file to track all runs
LOG_FILE="$BASE_DIR/training_experiments_log.txt"
echo "Starting training experiments at $(date)" > "$LOG_FILE"
echo "===========================================" >> "$LOG_FILE"
echo "Configuration Constants:" >> "$LOG_FILE"
echo "- Batch Size: 8" >> "$LOG_FILE"
echo "- Gradient Accumulation Steps: 8 (effective batch size: 64)" >> "$LOG_FILE"
echo "- Transformer Heads: 8" >> "$LOG_FILE"
echo "- Transformer Layers: 4" >> "$LOG_FILE"
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
# SECTION A: Learning Rate Variations
# ============================================================

# ============================================================
# Experiment A1: Binary with very small learning rate (1e-4)
# ============================================================
run_experiment "binary_very_small_lr" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_very_small_lr"

# ============================================================
# Experiment A2: Binary with small learning rate (3e-4)
# ============================================================
run_experiment "binary_small_lr" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 3e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_small_lr"

# ============================================================
# Experiment A3: Binary with medium learning rate (5e-4)
# ============================================================
run_experiment "binary_medium_lr" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_medium_lr"

# ============================================================
# Experiment A4: Valence-based with very small learning rate (1e-4)
# ============================================================
run_experiment "va5_very_small_lr" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_very_small_lr"

# ============================================================
# Experiment A5: Valence-based with small learning rate (3e-4)
# ============================================================
run_experiment "va5_small_lr" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 3e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_small_lr"

# ============================================================
# Experiment A6: Valence-based with medium learning rate (5e-4)
# ============================================================
run_experiment "va5_medium_lr" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_medium_lr"

# ============================================================
# Experiment A7: Arousal-based with very small learning rate (1e-4)
# ============================================================
run_experiment "ar5_very_small_lr" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 1e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_very_small_lr"

# ============================================================
# Experiment A8: Arousal-based with small learning rate (3e-4)
# ============================================================
run_experiment "ar5_small_lr" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 3e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_small_lr"

# ============================================================
# Experiment A9: Arousal-based with medium learning rate (5e-4)
# ============================================================
run_experiment "ar5_medium_lr" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_medium_lr"

# ============================================================
# SECTION B: Dropout Variations
# ============================================================

# ============================================================
# Experiment B1: Binary with low dropout (0.1)
# ============================================================
run_experiment "binary_low_dropout" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.1 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_low_dropout"

# ============================================================
# Experiment B2: Binary with medium dropout (0.2)
# ============================================================
run_experiment "binary_medium_dropout" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_medium_dropout"

# ============================================================
# Experiment B3: Binary with high dropout (0.3)
# ============================================================
run_experiment "binary_high_dropout" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.3 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_high_dropout"

# ============================================================
# Experiment B4: Valence-based with low dropout (0.1)
# ============================================================
run_experiment "va5_low_dropout" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.1 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_low_dropout"

# ============================================================
# Experiment B5: Valence-based with medium dropout (0.2)
# ============================================================
run_experiment "va5_medium_dropout" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_medium_dropout"

# ============================================================
# Experiment B6: Valence-based with high dropout (0.3)
# ============================================================
run_experiment "va5_high_dropout" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.3 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_high_dropout"

# ============================================================
# Experiment B7: Arousal-based with low dropout (0.1)
# ============================================================
run_experiment "ar5_low_dropout" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.1 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_low_dropout"

# ============================================================
# Experiment B8: Arousal-based with medium dropout (0.2)
# ============================================================
run_experiment "ar5_medium_dropout" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_medium_dropout"

# ============================================================
# Experiment B9: Arousal-based with high dropout (0.3)
# ============================================================
run_experiment "ar5_high_dropout" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.3 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_high_dropout"

# ============================================================
# SECTION C: Weight Decay Variations
# ============================================================

# ============================================================
# Experiment C1: Binary with lower weight decay (1e-6)
# ============================================================
run_experiment "binary_lower_wd" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --weight_decay 1e-6 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_lower_wd"

# ============================================================
# Experiment C2: Binary with higher weight decay (5e-5)
# ============================================================
run_experiment "binary_higher_wd" \
    --task binary \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --weight_decay 5e-5 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/binary_higher_wd"

# ============================================================
# Experiment C3: Valence-based with lower weight decay (1e-6)
# ============================================================
run_experiment "va5_lower_wd" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --weight_decay 1e-6 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_lower_wd"

# ============================================================
# Experiment C4: Valence-based with higher weight decay (5e-5)
# ============================================================
run_experiment "va5_higher_wd" \
    --task va5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --weight_decay 5e-5 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/va5_higher_wd"

# ============================================================
# Experiment C5: Arousal-based with lower weight decay (1e-6)
# ============================================================
run_experiment "ar5_lower_wd" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --weight_decay 1e-6 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_lower_wd"

# ============================================================
# Experiment C6: Arousal-based with higher weight decay (5e-5)
# ============================================================
run_experiment "ar5_higher_wd" \
    --task ar5 \
    --feature_dim 640 \
    --num_heads 8 \
    --num_layers 4 \
    --d_ff 2560 \
    --use_projection \
    --projection_dim 1280 \
    --classifier_hidden_dim 512 \
    --learning_rate 5e-4 \
    --dropout 0.2 \
    --weight_decay 5e-5 \
    --batch_size 8 \
    --use_gradient_accumulation \
    --accumulation_steps 8 \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
    --checkpoint_dir "$BASE_DIR/ar5_higher_wd"

echo "All training experiments completed!"
echo "Log file available at: $LOG_FILE" 