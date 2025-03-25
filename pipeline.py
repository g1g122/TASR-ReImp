"""
Author: Etlazure
Creation Date: March 7, 2025
Purpose: Complete pipeline for multi-modal emotion recognition with Transformer-based architecture
"""

import os
import time
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Import model components
from feature_extraction import MultiModalEmbedding
from encoder import TransformerEncoder
from classifier import EmotionClassifier, calculate_metrics

class MultiModalTransformer(nn.Module):
    """
    Complete multi-modal Transformer-based model for emotion recognition.
    Combines feature extraction, Transformer encoder, and classification head.
    """
    def __init__(self, config):
        """
        Initialize the complete model
        
        Parameters:
            config: Dictionary containing model hyperparameters
        """
        super(MultiModalTransformer, self).__init__()
        
        # Store configuration
        self.config = config
        
        # Feature extraction and embedding module
        self.feature_embedding = MultiModalEmbedding(
            feature_dim=config['feature_dim'],
            seq_len=config['seq_len'],
            use_projection=config['use_projection'],
            projection_dim=config['projection_dim'],
            dropout=config['dropout']
        )
        
        # Calculate input dimension for encoder
        d_model = self.feature_embedding.output_dim
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout']
        )
        
        # Classification head
        self.classifier = EmotionClassifier(
            input_dim=d_model,
            hidden_dim=config['classifier_hidden_dim'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
    def forward(self, hr, gsr, motion):
        """
        Forward pass through the complete model
        
        Parameters:
            hr: Heart rate data with shape [batch_size, 1, seq_len]
            gsr: Galvanic skin response data with shape [batch_size, 1, seq_len]
            motion: Motion data with shape [batch_size, 1, seq_len]
            
        Returns:
            logits: Raw scores for each class
        """
        # Extract and embed features
        embedded_features = self.feature_embedding(hr, gsr, motion)
        
        # Pass through Transformer encoder
        encoded_features, _ = self.transformer_encoder(embedded_features)
        
        # Classify encoded features
        logits = self.classifier(encoded_features)
        
        return logits
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PhysiologicalDataset(Dataset):
    """
    Dataset class for physiological data
    """
    def __init__(self, hr_data, gsr_data, motion_data, labels):
        """
        Initialize dataset
        
        Parameters:
            hr_data: Heart rate data tensor [num_samples, 1, seq_len]
            gsr_data: GSR data tensor [num_samples, 1, seq_len]
            motion_data: Motion/accelerometer data tensor [num_samples, 1, seq_len]
            labels: Labels tensor [num_samples]
        """
        self.hr_data = hr_data
        self.gsr_data = gsr_data
        self.motion_data = motion_data
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.hr_data[idx],
            self.gsr_data[idx],
            self.motion_data[idx],
            self.labels[idx]
        )

def load_dataset(data_path, task='va5', train_ratio=0.7, val_ratio=0.1, seed=42):
    """
    Load and preprocess the dataset, splitting by subjects to avoid cross-contamination
    
    Parameters:
        data_path: Path to the aligned_events.pkl file
        task: Classification task ('binary' for PANAS-based, 'va5' for valence-based 5-class, 
              or 'ar5' for arousal-based 5-class)
        train_ratio: Ratio of subjects to use for training (default: 0.7)
        val_ratio: Ratio of subjects to use for validation (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for testing data
    """
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        aligned_events = pickle.load(f)
    
    # Get list of subjects
    subjects = list(aligned_events.keys())
    np.random.seed(seed)
    np.random.shuffle(subjects)
    
    # Split subjects into train, validation and test sets
    n_subjects = len(subjects)
    train_subjects = subjects[:int(n_subjects * train_ratio)]
    val_subjects = subjects[int(n_subjects * train_ratio):int(n_subjects * (train_ratio + val_ratio))]
    test_subjects = subjects[int(n_subjects * (train_ratio + val_ratio)):]
    
    print(f"Training on {len(train_subjects)} subjects, validating on {len(val_subjects)} subjects, testing on {len(test_subjects)} subjects")
    
    # Process training data
    train_hr, train_gsr, train_motion, train_labels = process_subject_data(
        aligned_events, train_subjects, task)
    
    # Process validation data
    val_hr, val_gsr, val_motion, val_labels = process_subject_data(
        aligned_events, val_subjects, task)
    
    # Process test data
    test_hr, test_gsr, test_motion, test_labels = process_subject_data(
        aligned_events, test_subjects, task)
    
    # Create datasets
    train_dataset = PhysiologicalDataset(train_hr, train_gsr, train_motion, train_labels)
    val_dataset = PhysiologicalDataset(val_hr, val_gsr, val_motion, val_labels)
    test_dataset = PhysiologicalDataset(test_hr, test_gsr, test_motion, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")
    print(f"Testing data: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def process_subject_data(aligned_events, subjects, task):
    """
    Process data for a list of subjects
    
    Parameters:
        aligned_events: Dictionary containing aligned event data
        subjects: List of subject IDs to process
        task: Classification task ('binary' for PANAS-based, 'va5' for valence-based 5-class, 
              or 'ar5' for arousal-based 5-class)
    
    Returns:
        hr_data: Heart rate data tensor [num_samples, 1, seq_len]
        gsr_data: GSR data tensor [num_samples, 1, seq_len]
        motion_data: Motion data tensor [num_samples, 1, seq_len]
        labels: Labels tensor [num_samples]
    """
    hr_list, gsr_list, motion_list, label_list = [], [], [], []
    
    for subject in subjects:
        for event_id in aligned_events[subject].keys():
            event_data = aligned_events[subject][event_id]
            
            # Extract physiological data
            hr = event_data['heart_rate']
            gsr = event_data['GSR']
            motion = event_data['motion']
            
            # Make sure data is of expected length, truncate or pad if necessary
            seq_len = config['seq_len']
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
            hr_list.append(hr_normalized)
            gsr_list.append(gsr_normalized)
            motion_list.append(motion_normalized)
            label_list.append(label)
    
    # Convert to tensors
    hr_data = torch.FloatTensor(np.array(hr_list)).unsqueeze(1)
    gsr_data = torch.FloatTensor(np.array(gsr_list)).unsqueeze(1)
    motion_data = torch.FloatTensor(np.array(motion_list)).unsqueeze(1)
    labels = torch.LongTensor(np.array(label_list))
    
    return hr_data, gsr_data, motion_data, labels

def train_model(model, train_loader, val_loader, config):
    """
    Train the model
    
    Parameters:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary containing training hyperparameters
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_metrics: List of validation metrics
    """
    # Setup device
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(config['beta1'], config['beta2']),
        eps=config['epsilon']
    )
    
    # Linear learning rate scheduler
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=config['epochs']
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    val_metrics = []
    best_val_f1 = 0.0
    best_model_path = None
    patience_counter = 0
    
    # Create directory for checkpoints
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Gradient accumulation steps
    accumulation_steps = config['accumulation_steps']
    effective_batch_size = config['batch_size'] * accumulation_steps
    print(f"Using gradient accumulation with {accumulation_steps} steps")
    print(f"Effective batch size: {effective_batch_size}")
    
    print("Starting training...")
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        accumulated_samples = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}") as progress_bar:
            # Zero gradients at the beginning of each epoch
            optimizer.zero_grad()
            
            for batch_idx, (hr, gsr, motion, labels) in enumerate(progress_bar):
                # Move data to device
                hr, gsr, motion, labels = hr.to(device), gsr.to(device), motion.to(device), labels.to(device)
                
                # Forward pass
                logits = model(hr, gsr, motion)
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Normalize loss by accumulation steps (to keep loss scale consistent)
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update tracking (using unnormalized loss for reporting)
                batch_loss = loss.item() * accumulation_steps
                epoch_loss += batch_loss
                batch_count += 1
                accumulated_samples += labels.size(0)
                
                # Only update weights after accumulating enough gradients
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # Update weights
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Calculate effective samples processed
                    effective_samples = min(accumulated_samples, effective_batch_size)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": batch_loss, 
                        "acc_samples": f"{accumulated_samples}/{effective_batch_size}"
                    })
                    
                    # Reset accumulated samples
                    accumulated_samples = 0
                else:
                    progress_bar.set_postfix({
                        "loss": batch_loss, 
                        "acc_samples": f"{accumulated_samples}/{effective_batch_size}"
                    })
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for hr, gsr, motion, labels in val_loader:
                # Move data to device
                hr, gsr, motion, labels = hr.to(device), gsr.to(device), motion.to(device), labels.to(device)
                
                # Forward pass
                logits = model(hr, gsr, motion)
                
                # Calculate loss
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                # Store for metrics calculation
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_metrics_dict = calculate_metrics(
            torch.tensor(all_predictions), 
            torch.tensor(all_labels)
        )
        
        # Add average validation loss
        val_metrics_dict['loss'] = val_loss / len(val_loader)
        val_metrics.append(val_metrics_dict)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics_dict['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics_dict['accuracy']:.4f}")
        print(f"  Val Precision: {val_metrics_dict['precision']:.4f}")
        print(f"  Val F1 Score: {val_metrics_dict['f1_score']:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint if f1 score improves
        if val_metrics_dict['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics_dict['f1_score']
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"model_epoch_{epoch+1}.pth")
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics_dict,
                'config': config
            }, checkpoint_path)
            
            print(f"  Model checkpoint saved to {checkpoint_path}")
            best_model_path = checkpoint_path
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_path is not None:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with F1 score: {best_val_f1:.4f}")
    
    return model, train_losses, val_metrics

def test_model(model, test_loader, config):
    """
    Test the model on the test set
    
    Parameters:
        model: Trained model
        test_loader: DataLoader for test data
        config: Dictionary containing test hyperparameters
    
    Returns:
        test_metrics: Dictionary containing test metrics
    """
    # Setup device
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Test tracking
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    
    print("Starting testing...")
    with torch.no_grad():
        for hr, gsr, motion, labels in tqdm(test_loader, desc="Testing"):
            # Move data to device
            hr, gsr, motion, labels = hr.to(device), gsr.to(device), motion.to(device), labels.to(device)
            
            # Forward pass
            logits = model(hr, gsr, motion)
            
            # Calculate loss
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Store for metrics calculation
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate test metrics
    test_metrics = calculate_metrics(
        torch.tensor(all_predictions), 
        torch.tensor(all_labels)
    )
    
    # Add average test loss
    test_metrics['loss'] = test_loss / len(test_loader)
    
    # Print test results
    print("\nTest Results:")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test F1 Score: {test_metrics['f1_score']:.4f}")
    
    return test_metrics

def plot_training_history(train_losses, val_metrics, config):
    """
    Plot training history
    
    Parameters:
        train_losses: List of training losses
        val_metrics: List of validation metrics
        config: Dictionary containing plot configuration
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot([m['loss'] for m in val_metrics], label='Val Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot validation metrics
    ax2.plot([m['accuracy'] for m in val_metrics], label='Accuracy')
    ax2.plot([m['precision'] for m in val_metrics], label='Precision')
    ax2.plot([m['f1_score'] for m in val_metrics], label='F1 Score')
    ax2.set_title('Validation Metrics vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(config['checkpoint_dir'], 'training_history.png'))
    plt.close()

def print_config(config):
    """
    Print model and training configuration
    
    Parameters:
        config: Dictionary containing configuration
    """
    print("\n" + "="*80)
    print("MODEL CONFIGURATION")
    print("="*80)
    
    # Model architecture parameters
    print("\nModel Architecture:")
    print(f"  Feature Dimension: {config['feature_dim']}")
    print(f"  Use Projection: {config['use_projection']}")
    print(f"  Projection Dimension: {config['projection_dim'] if config['use_projection'] else 'N/A'}")
    print(f"  Number of Transformer Heads: {config['num_heads']}")
    print(f"  Number of Transformer Layers: {config['num_layers']}")
    print(f"  Feed-Forward Dimension: {config['d_ff']}")
    print(f"  Classifier Hidden Dimension: {config['classifier_hidden_dim']}")
    print(f"  Number of Classes: {config['num_classes']}")
    print(f"  Dropout Rate: {config['dropout']}")
    
    # Training parameters
    print("\nTraining Parameters:")
    print(f"  Task: {config['task']}")
    print(f"  Training Ratio: {config['train_ratio']}")
    print(f"  Validation Ratio: {config['val_ratio']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Gradient Accumulation Steps: {config['accumulation_steps']}")
    print(f"  Effective Batch Size: {config['batch_size'] * config['accumulation_steps']}")
    print(f"  Number of Epochs: {config['epochs']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Weight Decay: {config['weight_decay']}")
    print(f"  Beta1: {config['beta1']}")
    print(f"  Beta2: {config['beta2']}")
    print(f"  Epsilon: {config['epsilon']}")
    print(f"  Early Stopping Patience: {config['patience']}")
    print(f"  Device: {config['device']}")
    
    # Data parameters
    print("\nData Parameters:")
    print(f"  Data Path: {config['data_path']}")
    print(f"  Sequence Length: {config['seq_len']}")
    print(f"  Number of Workers: {config['num_workers']}")
    print(f"  Random Seed: {config['seed']}")
    
    # Output parameters
    print("\nOutput Parameters:")
    print(f"  Checkpoint Directory: {config['checkpoint_dir']}")
    print("="*80 + "\n")

def main():
    """Main function for model training and testing"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Multi-Modal Emotion Recognition')
    
    # Model architecture
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension for each modality')
    parser.add_argument('--use_projection', action='store_true', help='Whether to use projection layer')
    parser.add_argument('--projection_dim', type=int, default=None, help='Dimension after projection')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--classifier_hidden_dim', type=int, default=512, help='Classifier hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Task settings
    parser.add_argument('--task', type=str, default='va5', choices=['binary', 'va5', 'ar5'], 
                        help='Classification task (binary, va5: valence-based 5-class, or ar5: arousal-based 5-class)')
    
    # Training settings
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--accumulation_steps', type=int, default=8, 
                        help='Number of gradient accumulation steps (default: 8, giving effective batch size of 64)')
    parser.add_argument('--use_gradient_accumulation', action='store_true', 
                        help='Whether to use gradient accumulation (if not set, accumulation_steps=1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay coefficient')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    
    # Data settings
    parser.add_argument('--data_path', type=str, default='aligned_events.pkl', help='Path to data file')
    parser.add_argument('--seq_len', type=int, default=1800, help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output settings
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory for saving checkpoints')
    
    args = parser.parse_args()
    
    # Determine device
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    
    # Set accumulation steps based on args
    accumulation_steps = 1
    if args.use_gradient_accumulation:
        accumulation_steps = args.accumulation_steps
    
    # Create config dictionary
    global config
    config = {
        # Model architecture
        'feature_dim': args.feature_dim,
        'use_projection': args.use_projection,
        'projection_dim': args.projection_dim,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'd_ff': args.d_ff,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'num_classes': 2 if args.task == 'binary' else 5,
        'dropout': args.dropout,
        
        # Task settings
        'task': args.task,
        
        # Training settings
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'batch_size': args.batch_size,
        'accumulation_steps': accumulation_steps,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'epsilon': args.epsilon,
        'patience': args.patience,
        'device': device,
        
        # Data settings
        'data_path': args.data_path,
        'seq_len': args.seq_len,
        'num_workers': args.num_workers,
        'seed': args.seed,
        
        # Output settings
        'checkpoint_dir': args.checkpoint_dir,
    }
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Print configuration
    print_config(config)
    
    # Add timestamp to checkpoint directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config['checkpoint_dir'] = os.path.join(
        config['checkpoint_dir'], 
        f"{config['task']}_{timestamp}"
    )
    
    # Load data
    train_loader, val_loader, test_loader = load_dataset(
        config['data_path'], 
        task=config['task'], 
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    # Initialize model
    model = MultiModalTransformer(config)
    
    # Print model summary
    print(f"\nModel has {model.count_parameters():,} trainable parameters")
    
    # Train model
    model, train_losses, val_metrics = train_model(
        model, 
        train_loader, 
        val_loader,  # Using validation_loader for validation
        config
    )
    
    # Test model
    test_metrics = test_model(model, test_loader, config)
    
    # Plot training history
    plot_training_history(train_losses, val_metrics, config)
    
    # Save final results
    results = {
        'config': config,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    with open(os.path.join(config['checkpoint_dir'], 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {os.path.join(config['checkpoint_dir'], 'results.pkl')}")

if __name__ == "__main__":
    main() 