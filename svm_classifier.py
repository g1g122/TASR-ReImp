"""
Author: Etlazure
Creation Date: March 18, 2025
Purpose: Support Vector Machine (SVM) implementation for multi-modal emotion recognition
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_dataset(data_path, task='va5', train_ratio=0.8, seed=42):
    """
    Load and preprocess the dataset, splitting by subjects to avoid cross-contamination
    
    Parameters:
        data_path: Path to the aligned_events.pkl file
        task: Classification task ('binary' for PANAS-based, 'va5' for valence-based 5-class, 
              or 'ar5' for arousal-based 5-class)
        train_ratio: Ratio of subjects to use for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
    """
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        aligned_events = pickle.load(f)
    
    # Get list of subjects
    subjects = list(aligned_events.keys())
    np.random.seed(seed)
    np.random.shuffle(subjects)
    
    # Split subjects into train and test sets (80:20 ratio as specified in the article)
    n_subjects = len(subjects)
    train_subjects = subjects[:int(n_subjects * train_ratio)]
    test_subjects = subjects[int(n_subjects * train_ratio):]
    
    print(f"Training on {len(train_subjects)} subjects, testing on {len(test_subjects)} subjects")
    
    # Process training data
    X_train, y_train = process_subject_data(aligned_events, train_subjects, task)
    
    # Process test data
    X_test, y_test = process_subject_data(aligned_events, test_subjects, task)
    
    print(f"Training data: {len(y_train)} samples")
    print(f"Testing data: {len(y_test)} samples")
    
    return X_train, y_train, X_test, y_test

def process_subject_data(aligned_events, subjects, task, seq_len=1800):
    """
    Process data for a list of subjects
    
    Parameters:
        aligned_events: Dictionary containing aligned event data
        subjects: List of subject IDs to process
        task: Classification task ('binary' for PANAS-based, 'va5' for valence-based 5-class, 
              or 'ar5' for arousal-based 5-class)
        seq_len: Length of the sequence to extract
    
    Returns:
        features: Feature matrix [num_samples, num_features]
        labels: Labels array [num_samples]
    """
    features_list = []
    label_list = []
    
    for subject in subjects:
        for event_id in aligned_events[subject].keys():
            event_data = aligned_events[subject][event_id]
            
            # Extract physiological data
            hr = event_data['heart_rate']
            gsr = event_data['GSR']
            motion = event_data['motion']
            
            # Make sure data is of expected length, truncate or pad if necessary
            if len(hr) > seq_len:
                hr = hr[:seq_len]
                gsr = gsr[:seq_len]
                motion = motion[:seq_len]
            elif len(hr) < seq_len:
                # Pad with zeros
                hr = np.pad(hr, (0, seq_len - len(hr)), 'constant')
                gsr = np.pad(gsr, (0, seq_len - len(gsr)), 'constant')
                motion = np.pad(motion, (0, seq_len - len(motion)), 'constant')
            
            # Normalize data for each subject separately to avoid cross-subject information leakage
            hr_normalized = (hr - np.mean(hr)) / (np.std(hr) + 1e-8)
            gsr_normalized = (gsr - np.mean(gsr)) / (np.std(gsr) + 1e-8)
            motion_normalized = (motion - np.mean(motion)) / (np.std(motion) + 1e-8)
            
            # Extract statistical features from each modality
            hr_features = extract_statistical_features(hr_normalized)
            gsr_features = extract_statistical_features(gsr_normalized)
            motion_features = extract_statistical_features(motion_normalized)
            
            # Combine features
            combined_features = np.concatenate([hr_features, gsr_features, motion_features])
            
            # Get labels based on task
            if task == 'binary':
                # Binary classification based on PANAS scores
                panas = event_data['panas']
                
                # Positive affect items (indices 4, 6, 7, 9)
                positive_indices = [4, 6, 7, 9]
                positive_score = np.sum(panas[positive_indices])
                
                # Negative affect items (indices 0, 1, 2, 3, 5, 8)
                negative_indices = [0, 1, 2, 3, 5, 8]
                negative_score = np.sum(panas[negative_indices])
                
                # Assign label based on which score has the higher value
                label = 1 if positive_score > negative_score else 0
            elif task == 'va5':
                # 5-class classification based on valence scores (1-5)
                # Valence score directly maps to class (score 1 = class 0, score 5 = class 4)
                valence = event_data['valence']
                
                # Convert 1-5 scale to 0-4 class index
                label = int(valence) - 1
                
                # Validate label is in the correct range
                if label < 0:
                    label = 0
                elif label > 4:
                    label = 4
            elif task == 'ar5':
                # 5-class classification based on arousal scores (1-5)
                # Arousal score directly maps to class (score 1 = class 0, score 5 = class 4)
                arousal = event_data['arousal']
                
                # Convert 1-5 scale to 0-4 class index
                label = int(arousal) - 1
                
                # Validate label is in the correct range
                if label < 0:
                    label = 0
                elif label > 4:
                    label = 4
            
            # Add to lists
            features_list.append(combined_features)
            label_list.append(label)
    
    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(label_list)
    
    return features, labels

def extract_statistical_features(signal):
    """
    Extract statistical features from time series data
    
    Parameters:
        signal: Time series data
        
    Returns:
        features: Statistical features array
    """
    # Statistical features
    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    range_val = max_val - min_val
    median = np.median(signal)
    q25 = np.percentile(signal, 25)
    q75 = np.percentile(signal, 75)
    iqr = q75 - q25
    skewness = np.mean(((signal - mean) / (std + 1e-8)) ** 3) if std > 0 else 0
    kurtosis = np.mean(((signal - mean) / (std + 1e-8)) ** 4) - 3 if std > 0 else 0
    
    # Frequency domain features (using FFT)
    try:
        fft_vals = np.abs(np.fft.rfft(signal))
        fft_freq = np.fft.rfftfreq(len(signal))
        
        # Use only first 10 frequency components
        fft_features = fft_vals[:10] if len(fft_vals) >= 10 else np.pad(fft_vals, (0, 10 - len(fft_vals)), 'constant')
    except:
        fft_features = np.zeros(10)
    
    # Combine features
    features = np.array([
        mean, std, max_val, min_val, range_val, median, 
        q25, q75, iqr, skewness, kurtosis
    ])
    
    # Add frequency features
    features = np.concatenate([features, fft_features])
    
    return features

def calculate_metrics(y_true, y_pred):
    """
    Calculate various classification metrics
    
    Parameters:
        y_true: Ground truth labels
        y_pred: Model predictions
        
    Returns:
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    # For multi-class classification, use macro averaging
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def plot_confusion_matrix(y_true, y_pred, task, save_path=None):
    """
    Plot confusion matrix
    
    Parameters:
        y_true: Ground truth labels
        y_pred: Model predictions
        task: Classification task ('binary', 'va5', or 'ar5')
        save_path: Path to save the plot (if None, the plot is shown)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Set labels based on task
    if task == 'binary':
        class_labels = ['Negative', 'Positive']
    elif task == 'va5':
        class_labels = ['Very Low', 'Low', 'Neutral', 'High', 'Very High']
    elif task == 'ar5':
        class_labels = ['Very Low', 'Low', 'Neutral', 'High', 'Very High']
    else:
        class_labels = [str(i) for i in range(len(np.unique(y_true)))]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {task.upper()} Task')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    """Main function for SVM training and testing"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='SVM Emotion Recognition')
    
    # Task settings
    parser.add_argument('--task', type=str, default='va5', choices=['binary', 'va5', 'ar5'], 
                        help='Classification task (binary, va5: valence-based 5-class, or ar5: arousal-based 5-class)')
    
    # Data settings
    parser.add_argument('--data_path', type=str, default='aligned_events.pkl', help='Path to data file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train split ratio (following 80:20 as in the article)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output settings
    parser.add_argument('--results_dir', type=str, default='svm_results', 
                        help='Directory for saving results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(
        args.data_path, 
        task=args.task, 
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Print dataset statistics
    print(f"\nFeature shape: {X_train.shape} (training), {X_test.shape} (testing)")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Create SVM pipeline with StandardScaler and SVM
    # Using RBF kernel, C=1.0, gamma=0.1 as specified in the article
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma=0.1, random_state=args.seed))
    ])
    
    # Train SVM
    print("\nTraining SVM model...")
    svm_pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = svm_pipeline.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Print results
    print("\nTest Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Plot confusion matrix
    confusion_matrix_path = os.path.join(args.results_dir, f'confusion_matrix_{args.task}.png')
    plot_confusion_matrix(y_test, y_pred, args.task, save_path=confusion_matrix_path)
    print(f"Confusion matrix saved to {confusion_matrix_path}")
    
    # Save results
    results = {
        'args': vars(args),
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    results_path = os.path.join(args.results_dir, f'results_{args.task}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main() 