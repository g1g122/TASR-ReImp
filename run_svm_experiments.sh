#!/bin/bash

# Script to run SVM experiments for all classification tasks

# Create timestamp for results directory
RESULTS_DIR="svm_results"

echo "Creating results directory: ${RESULTS_DIR}"
mkdir -p ${RESULTS_DIR}

# Log file
LOG_FILE="${RESULTS_DIR}/svm_experiments.log"
touch ${LOG_FILE}

# Function to run experiment and log results
run_experiment() {
    task=$1
    
    echo "----------------------------------------" | tee -a ${LOG_FILE}
    echo "Running SVM experiment for task: ${task}" | tee -a ${LOG_FILE}
    echo "----------------------------------------" | tee -a ${LOG_FILE}
    
    # Run SVM classifier with specified task
    python svm_classifier.py --task ${task} --results_dir ${RESULTS_DIR} | tee -a ${LOG_FILE}
    
    echo "Experiment completed for task: ${task}" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
}

# Run binary classification experiment
echo "Starting binary classification experiment..." | tee -a ${LOG_FILE}
run_experiment "binary"

# Run valence-based 5-class classification experiment
echo "Starting valence-based 5-class classification experiment..." | tee -a ${LOG_FILE}
run_experiment "va5"

# Run arousal-based 5-class classification experiment
echo "Starting arousal-based 5-class classification experiment..." | tee -a ${LOG_FILE}
run_experiment "ar5"

echo "All experiments completed successfully!" | tee -a ${LOG_FILE}
echo "Results saved to: ${RESULTS_DIR}" | tee -a ${LOG_FILE} 