"""
Author: Etlazure
Creation Date: March 4, 2025
Purpose: Implement the classification head for multi-modal emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmotionClassifier(nn.Module):
    """
    Classification head for emotion recognition.
    Takes encoded features from TransformerEncoder and outputs emotion predictions.
    Supports both binary classification (based on PANAS scores) and 
    5-class classification (based on valence and arousal scores).
    """
    def __init__(self, input_dim=1536, hidden_dim=512, num_classes=5, dropout=0.2):
        """
        Initialize the classifier
        
        Parameters:
            input_dim: Dimension of input features from encoder (default: 1536)
            hidden_dim: Dimension of hidden layer (default: 512)
            num_classes: Number of output classes (default: 5 for VA-based classification)
            dropout: Dropout probability (default: 0.2)
        """
        super(EmotionClassifier, self).__init__()
        
        # Two-layer feed-forward network with ReLU activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
            x: Input tensor with shape [batch_size, input_dim]
               This should be the output from TransformerEncoder
               
        Returns:
            logits: Raw scores with shape [batch_size, num_classes]
        """
        # First layer with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer (logits)
        logits = self.fc2(x)
        return logits

def calculate_metrics(predictions, targets):
    """
    Calculate various classification metrics
    
    Parameters:
        predictions: Model predictions (after argmax)
        targets: Ground truth labels
        
    Returns:
        Dictionary containing accuracy, precision, and F1 score
    """
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    
    # For multi-class, calculate macro precision and F1
    precision_list = []
    f1_list = []
    
    for cls in range(targets.max().item() + 1):
        true_positives = ((predictions == cls) & (targets == cls)).sum().float()
        predicted_positives = (predictions == cls).sum().float()
        actual_positives = (targets == cls).sum().float()
        
        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        precision_list.append(precision.item())
        f1_list.append(f1.item())
    
    return {
        'accuracy': accuracy,
        'precision': np.mean(precision_list),
        'f1_score': np.mean(f1_list)
    }

# Test code
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test the classifier with synthetic data
    batch_size = 16
    input_dim = 1536
    hidden_dim = 512
    num_classes = 5
    
    # Create random input and labels
    x = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Initialize classifier
    classifier = EmotionClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    
    # Forward pass
    logits = classifier(x)
    
    # Get predictions
    predictions = torch.argmax(logits, dim=1)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    
    # Print results
    print("Test Results:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    print("\nTest completed successfully!") 