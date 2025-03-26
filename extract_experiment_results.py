import re
import pandas as pd
import os

# Path to log file
# log_file_path = 'experiment_results/architecture_experiments_log.txt'
# log_file_path = 'experiment_results/projection_experiments_log.txt'
log_file_path = 'experiment_results/training_experiments_log.txt'

# Define columns for the DataFrame
columns = [
    'Experiment', 'Feature Dimension', 'Use Projection', 'Projection Dimension',
    'Number of Transformer Heads', 'Number of Transformer Layers', 'Feed-Forward Dimension',
    'Classifier Hidden Dimension', 'Number of Classes', 'Dropout Rate', 'Task',
    'Training Ratio', 'Validation Ratio', 'Batch Size', 'Gradient Accumulation Steps',
    'Effective Batch Size', 'Number of Epochs', 'Learning Rate', 'Weight Decay',
    'Beta1', 'Beta2', 'Epsilon', 'Early Stopping Patience', 'Test Loss',
    'Test Accuracy', 'Test Precision', 'Test F1 Score'
]

# Create a list to store extracted data
experiments_data = []

# Regular expressions for extracting data
experiment_pattern = re.compile(r'Experiment: (.+)')
feature_dim_pattern = re.compile(r'Feature Dimension: (\d+)')
use_projection_pattern = re.compile(r'Use Projection: (\w+)')
projection_dim_pattern = re.compile(r'Projection Dimension: (.+)')
num_heads_pattern = re.compile(r'Number of Transformer Heads: (\d+)')
num_layers_pattern = re.compile(r'Number of Transformer Layers: (\d+)')
d_ff_pattern = re.compile(r'Feed-Forward Dimension: (\d+)')
classifier_hidden_dim_pattern = re.compile(r'Classifier Hidden Dimension: (\d+)')
num_classes_pattern = re.compile(r'Number of Classes: (\d+)')
dropout_pattern = re.compile(r'Dropout Rate: ([\d\.]+)')
task_pattern = re.compile(r'Task: (.+)')
train_ratio_pattern = re.compile(r'Training Ratio: ([\d\.]+)')
val_ratio_pattern = re.compile(r'Validation Ratio: ([\d\.]+)')
batch_size_pattern = re.compile(r'Batch Size: (\d+)')
accumulation_steps_pattern = re.compile(r'Gradient Accumulation Steps: (\d+)')
effective_batch_size_pattern = re.compile(r'Effective Batch Size: (\d+)')
num_epochs_pattern = re.compile(r'Number of Epochs: (\d+)')
lr_pattern = re.compile(r'Learning Rate: ([\d\.e\-]+)')
weight_decay_pattern = re.compile(r'Weight Decay: ([\d\.e\-]+)')
beta1_pattern = re.compile(r'Beta1: ([\d\.]+)')
beta2_pattern = re.compile(r'Beta2: ([\d\.]+)')
epsilon_pattern = re.compile(r'Epsilon: ([\d\.e\-]+)')
patience_pattern = re.compile(r'Early Stopping Patience: (\d+)')
test_loss_pattern = re.compile(r'Test Loss: ([\d\.]+)')
test_accuracy_pattern = re.compile(r'Test Accuracy: ([\d\.]+)')
test_precision_pattern = re.compile(r'Test Precision: ([\d\.]+)')
test_f1_pattern = re.compile(r'Test F1 Score: ([\d\.]+)')

# Read the log file
with open(log_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Split the content by experiment sections
experiment_sections = re.split(r'=+\s*\n\s*Experiment:', content)

# Skip the first element if it's empty or doesn't contain an experiment
if experiment_sections[0].strip() == '' or 'Experiment:' not in experiment_sections[0]:
    experiment_sections = experiment_sections[1:]
else:
    # If the first element contains an experiment, prepend "Experiment:" to it
    experiment_sections[0] = 'Experiment:' + experiment_sections[0]

# Process each experiment section
for section in experiment_sections:
    section = 'Experiment:' + section if not section.startswith('Experiment:') else section
    
    # Initialize a dictionary for the current experiment
    experiment_data = {col: None for col in columns}
    
    # Extract experiment name
    experiment_match = experiment_pattern.search(section)
    if experiment_match:
        experiment_data['Experiment'] = experiment_match.group(1).strip()
    
    # Extract model architecture parameters
    feature_dim_match = feature_dim_pattern.search(section)
    if feature_dim_match:
        experiment_data['Feature Dimension'] = int(feature_dim_match.group(1))
    
    use_projection_match = use_projection_pattern.search(section)
    if use_projection_match:
        experiment_data['Use Projection'] = use_projection_match.group(1).strip()
    
    projection_dim_match = projection_dim_pattern.search(section)
    if projection_dim_match:
        projection_dim = projection_dim_match.group(1).strip()
        experiment_data['Projection Dimension'] = projection_dim if projection_dim != 'N/A' else None
    
    num_heads_match = num_heads_pattern.search(section)
    if num_heads_match:
        experiment_data['Number of Transformer Heads'] = int(num_heads_match.group(1))
    
    num_layers_match = num_layers_pattern.search(section)
    if num_layers_match:
        experiment_data['Number of Transformer Layers'] = int(num_layers_match.group(1))
    
    d_ff_match = d_ff_pattern.search(section)
    if d_ff_match:
        experiment_data['Feed-Forward Dimension'] = int(d_ff_match.group(1))
    
    classifier_hidden_dim_match = classifier_hidden_dim_pattern.search(section)
    if classifier_hidden_dim_match:
        experiment_data['Classifier Hidden Dimension'] = int(classifier_hidden_dim_match.group(1))
    
    num_classes_match = num_classes_pattern.search(section)
    if num_classes_match:
        experiment_data['Number of Classes'] = int(num_classes_match.group(1))
    
    dropout_match = dropout_pattern.search(section)
    if dropout_match:
        experiment_data['Dropout Rate'] = float(dropout_match.group(1))
    
    # Extract training parameters
    task_match = task_pattern.search(section)
    if task_match:
        experiment_data['Task'] = task_match.group(1).strip()
    
    train_ratio_match = train_ratio_pattern.search(section)
    if train_ratio_match:
        experiment_data['Training Ratio'] = float(train_ratio_match.group(1))
    
    val_ratio_match = val_ratio_pattern.search(section)
    if val_ratio_match:
        experiment_data['Validation Ratio'] = float(val_ratio_match.group(1))
    
    batch_size_match = batch_size_pattern.search(section)
    if batch_size_match:
        experiment_data['Batch Size'] = int(batch_size_match.group(1))
    
    accumulation_steps_match = accumulation_steps_pattern.search(section)
    if accumulation_steps_match:
        experiment_data['Gradient Accumulation Steps'] = int(accumulation_steps_match.group(1))
    
    effective_batch_size_match = effective_batch_size_pattern.search(section)
    if effective_batch_size_match:
        experiment_data['Effective Batch Size'] = int(effective_batch_size_match.group(1))
    
    num_epochs_match = num_epochs_pattern.search(section)
    if num_epochs_match:
        experiment_data['Number of Epochs'] = int(num_epochs_match.group(1))
    
    lr_match = lr_pattern.search(section)
    if lr_match:
        experiment_data['Learning Rate'] = float(lr_match.group(1))
    
    weight_decay_match = weight_decay_pattern.search(section)
    if weight_decay_match:
        experiment_data['Weight Decay'] = float(weight_decay_match.group(1))
    
    beta1_match = beta1_pattern.search(section)
    if beta1_match:
        experiment_data['Beta1'] = float(beta1_match.group(1))
    
    beta2_match = beta2_pattern.search(section)
    if beta2_match:
        experiment_data['Beta2'] = float(beta2_match.group(1))
    
    epsilon_match = epsilon_pattern.search(section)
    if epsilon_match:
        experiment_data['Epsilon'] = float(epsilon_match.group(1))
    
    patience_match = patience_pattern.search(section)
    if patience_match:
        experiment_data['Early Stopping Patience'] = int(patience_match.group(1))
    
    # Extract test results
    test_loss_match = test_loss_pattern.search(section)
    if test_loss_match:
        experiment_data['Test Loss'] = float(test_loss_match.group(1))
    
    test_accuracy_match = test_accuracy_pattern.search(section)
    if test_accuracy_match:
        experiment_data['Test Accuracy'] = float(test_accuracy_match.group(1))
    
    test_precision_match = test_precision_pattern.search(section)
    if test_precision_match:
        experiment_data['Test Precision'] = float(test_precision_match.group(1))
    
    test_f1_match = test_f1_pattern.search(section)
    if test_f1_match:
        experiment_data['Test F1 Score'] = float(test_f1_match.group(1))
    
    # Add the experiment data to the list
    experiments_data.append(experiment_data)

# Create a DataFrame from the extracted data
df = pd.DataFrame(experiments_data, columns=columns)

# Create output directory if it doesn't exist
os.makedirs('experiment_results', exist_ok=True)

# Save the DataFrame to an Excel file
output_file = 'experiment_results/experiment_summary.xlsx'
df.to_excel(output_file, index=False)

print(f"Extracted {len(df)} experiments and saved to {output_file}") 