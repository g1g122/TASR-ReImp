"""
Author: Etlazure
Creation Date: February 28, 2025
Purpose: Create a feature extraction and embedding module for multimodal emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):
    """
    CNN feature extractor for physiological signals, preserving the time dimension.
    Uses LayerNorm instead of BatchNorm to avoid information leakage between subjects.
    According to the original paper, only one hidden layer (size 128) is used, with output dimension of 512.
    """
    def __init__(self, input_channels=1, hidden_size=128, output_dim=512, kernel_size=5):
        """
        Initialize the feature extractor
        
        Parameters:
            input_channels: Number of input channels, default is 1 (single channel time series)
            hidden_size: Hidden layer size, default is 128
            output_dim: Output feature dimension, default is 512
        """
        super(FeatureExtractor, self).__init__()
        
        # Define convolutional layers - using small kernels and same padding to preserve temporal information
        self.conv1 = nn.Conv1d(input_channels, hidden_size, kernel_size=kernel_size, stride=1, padding='same')
        self.conv2 = nn.Conv1d(hidden_size, output_dim, kernel_size=kernel_size, stride=1, padding='same')
        
        # Add learnable parameters for LayerNorm
        self.ln1_weight = nn.Parameter(torch.ones(hidden_size))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.ln2_weight = nn.Parameter(torch.ones(output_dim))
        self.ln2_bias = nn.Parameter(torch.zeros(output_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward propagation function
        
        Parameters:
            x: Input tensor with shape [batch_size, input_channels, seq_len]
            
        Returns:
            Feature sequence with shape [batch_size, output_dim, seq_len]
        """
        # First layer: Conv + LayerNorm + ReLU
        x = self.conv1(x)
        x = self.channel_layernorm(x, self.ln1_weight, self.ln1_bias)
        x = F.relu(x)
        
        # Second layer: Conv + LayerNorm + ReLU
        x = self.conv2(x)
        x = self.channel_layernorm(x, self.ln2_weight, self.ln2_bias)
        x = F.relu(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x
    
    def channel_layernorm(self, x, weight, bias, eps=1e-5):
        """
        Apply LayerNorm across the channel dimension for convolutional features, with learnable parameters
        
        Parameters:
            x: Input tensor with shape [batch_size, channels, seq_len]
            weight: Learnable scale parameter with shape [channels]
            bias: Learnable offset parameter with shape [channels]
            eps: Small constant to prevent division by zero
            
        Returns:
            Normalized tensor with the same shape
        """
        # Apply LayerNorm separately for each channel, while maintaining independence of batch and time dimensions
        # Calculate mean and variance (for each channel of each sample)
        mean = x.mean(dim=2, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + eps)
        
        # Apply learnable scale and offset
        # Expand weight and bias to match tensor dimensions [channels] -> [1, channels, 1]
        weight = weight.view(1, -1, 1)
        bias = bias.view(1, -1, 1)
        
        return x * weight + bias

class ParallelCrossAttention(nn.Module):
    """
    Parallel Cross-Attention module for multimodal feature fusion.
    Each modality acts as a query and attends to other modalities separately,
    then aggregates the results.
    """
    def __init__(self, feature_dim, num_modalities=3, dropout=0.2):
        """
        Initialize the parallel cross-attention module
        
        Parameters:
            feature_dim: Dimension of the feature vectors
            num_modalities: Number of modalities to fuse
            dropout: Dropout probability
        """
        super(ParallelCrossAttention, self).__init__()
        
        self.num_modalities = num_modalities
        
        # Create projections for each modality
        self.query_projs = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)])
        self.key_projs = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)])
        self.value_projs = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(num_modalities)])
        
        # Output projection to combine attended features
        self.output_proj = nn.Linear(feature_dim * num_modalities, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = feature_dim ** 0.5
        
    def forward(self, modality_features, return_attention=False):
        """
        Forward pass of parallel cross-attention
        
        Parameters:
            modality_features: List of modality features, each with shape [batch_size, seq_len, feature_dim]
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            List of attended features for each modality, each with shape [batch_size, seq_len, feature_dim]
            If return_attention is True, also returns attention weights dictionary
        """
        attended_features = []
        attention_weights = {} if return_attention else None
        
        for i, query_modality in enumerate(modality_features):
            # Project query
            q = self.query_projs[i](query_modality)
            
            # Initialize list to store attended features from each other modality
            attended_from_others = []
            
            # Store attention weights if requested
            if return_attention:
                attention_weights[f'modality_{i}'] = {}
            
            # Attend to each other modality
            for j, key_modality in enumerate(modality_features):
                if i != j:  # Skip self-attention
                    # Project key and value
                    k = self.key_projs[j](key_modality)
                    v = self.value_projs[j](key_modality)
                    
                    # Compute attention scores
                    scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
                    attn = F.softmax(scores, dim=-1)
                    attn = self.dropout(attn)
                    
                    # Store attention weights if requested
                    if return_attention:
                        attention_weights[f'modality_{i}'][f'modality_{j}'] = attn
                    
                    # Apply attention to values
                    attended = torch.matmul(attn, v)
                    attended_from_others.append(attended)
            
            # If there are no other modalities (should not happen in this case)
            if not attended_from_others:
                attended_features.append(query_modality)
                continue
            
            # Concatenate with original query features
            all_attended = [query_modality] + attended_from_others
            combined = torch.cat(all_attended, dim=-1)
            
            # Project back to feature dimension
            output = self.output_proj(combined)
            
            attended_features.append(output)
        
        if return_attention:
            return attended_features, attention_weights
        return attended_features

class PositionalEncoding(nn.Module):
    """
    Positional encoding to provide position information of elements in a sequence to the Transformer.
    Implemented according to the formula in the original paper.
    """
    def __init__(self, d_model, max_len=1800):
        """
        Initialize positional encoding
        
        Parameters:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # According to the original formula: P(i,2j) = sin(i/10000^(2j/d)), P(i,2j+1) = cos(i/10000^(2j/d))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine function to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine function to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer, not a model parameter, but will be saved and loaded
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Parameters:
            x: Input embeddings with shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Encoded embeddings with the same shape
        """
        return x + self.pe[:, :x.size(1)]

class MultiModalEmbedding(nn.Module):
    """
    Multi-modal embedding that uses cross-attention to fuse features from different modalities
    and adds positional encoding
    """
    def __init__(self, feature_dim=512, seq_len=1800, use_projection=False, projection_dim=None, dropout=0.2):
        """
        Initialize multi-modal embedding
        
        Parameters:
            feature_dim: Feature dimension for each modality, default is 512
            seq_len: Sequence length
            use_projection: Whether to use linear projection layer, default is False
            projection_dim: Dimension after projection, if None and use_projection is True, uses feature_dim*3
            dropout: Dropout probability
        """
        super(MultiModalEmbedding, self).__init__()
        
        # Create feature extractors for each modality
        self.hr_extractor = FeatureExtractor(input_channels=1, output_dim=feature_dim)
        self.gsr_extractor = FeatureExtractor(input_channels=1, output_dim=feature_dim)
        self.motion_extractor = FeatureExtractor(input_channels=1, output_dim=feature_dim)
        
        # Define cross-attention for modality fusion
        self.cross_attention = ParallelCrossAttention(feature_dim, num_modalities=3, dropout=dropout)
        
        # Calculate concatenated feature dimension
        concat_dim = feature_dim * 3
        
        # Decide whether to use projection layer
        self.use_projection = use_projection
        
        if use_projection:
            # If projection_dim is not specified, default to the same as concat_dim
            if projection_dim is None:
                projection_dim = concat_dim
                
            self.projection = nn.Linear(concat_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.output_dim = concat_dim
        
        # Create positional encoding using the actual output dimension
        self.positional_encoding = PositionalEncoding(d_model=self.output_dim, max_len=seq_len)
        
        # Store parameters
        self.seq_len = seq_len
        self.feature_dim = feature_dim
    
    def forward(self, hr, gsr, motion, return_attention=False):
        """
        Forward propagation function with parallel cross-attention between modalities
        
        Parameters:
            hr: Heart rate data with shape [batch_size, 1, seq_len]
            gsr: Galvanic skin response data with shape [batch_size, 1, seq_len]
            motion: Motion data with shape [batch_size, 1, seq_len]
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            Concatenated and encoded features with shape [batch_size, seq_len, output_dim]
            If return_attention is True, also returns attention weights dictionary
        """
        # Extract features - shape [batch_size, feature_dim, seq_len]
        hr_features = self.hr_extractor(hr)      
        gsr_features = self.gsr_extractor(gsr)   
        motion_features = self.motion_extractor(motion)
        
        # Transpose to match Transformer input format [batch_size, seq_len, feature_dim]
        hr_features = hr_features.permute(0, 2, 1)
        gsr_features = gsr_features.permute(0, 2, 1)
        motion_features = motion_features.permute(0, 2, 1)
        
        # Apply parallel cross-attention to fuse modalities
        # Each modality attends to the other modalities separately
        if return_attention:
            attended_features, attention_weights = self.cross_attention(
                [hr_features, gsr_features, motion_features], return_attention=True)
            hr_attended, gsr_attended, motion_attended = attended_features
        else:
            hr_attended, gsr_attended, motion_attended = self.cross_attention(
                [hr_features, gsr_features, motion_features])
        
        # Concatenate attended features
        concat_features = torch.cat([hr_attended, gsr_attended, motion_attended], dim=2)
        
        # Apply projection based on settings
        if self.use_projection:
            features = self.projection(concat_features)
        else:
            features = concat_features
        
        # Apply positional encoding
        encoded_features = self.positional_encoding(features)
        
        if return_attention:
            return encoded_features, attention_weights
        return encoded_features

def load_and_preprocess_data(data_path, participant_id=None, event_id=None):
    """
    Load and preprocess data
    
    Parameters:
        data_path: Path to aligned_events.pkl file
        participant_id: Specified participant ID, if None uses the first participant
        event_id: Specified event ID, if None uses the first event for that participant
        
    Returns:
        Tensors for heart rate, GSR, and motion data, as well as labels (emotional valence and arousal)
    """
    # Load data
    with open(data_path, 'rb') as f:
        aligned_events = pickle.load(f)
    
    # If participant ID is not specified, use the first participant
    if participant_id is None:
        participant_id = next(iter(aligned_events))
    
    # If event ID is not specified, use the first event for that participant
    if event_id is None:
        event_id = next(iter(aligned_events[participant_id]))
    
    # Get data
    event_data = aligned_events[participant_id][event_id]
    
    # Extract physiological data
    hr = event_data['heart_rate']
    gsr = event_data['GSR']
    motion = event_data['motion']
    
    # Extract labels
    valence = int(event_data['valence'])
    arousal = int(event_data['arousal'])
    panas = event_data['panas'].astype(int)
    
    # Normalize data - normalize each subject's data separately to avoid cross-subject information leakage
    hr_normalized = (hr - np.mean(hr)) / np.std(hr)
    gsr_normalized = (gsr - np.mean(gsr)) / np.std(gsr)
    motion_normalized = (motion - np.mean(motion)) / np.std(motion)
    
    # Convert to PyTorch tensors
    hr_tensor = torch.FloatTensor(hr_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
    gsr_tensor = torch.FloatTensor(gsr_normalized).unsqueeze(0).unsqueeze(0)
    motion_tensor = torch.FloatTensor(motion_normalized).unsqueeze(0).unsqueeze(0)
    
    # Create label tensors
    valence_tensor = torch.tensor(valence)
    arousal_tensor = torch.tensor(arousal)
    panas_tensor = torch.FloatTensor(panas)
    
    return hr_tensor, gsr_tensor, motion_tensor, valence_tensor, arousal_tensor, panas_tensor

# Test code
if __name__ == "__main__":
    # Set random seed to ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test data path
    data_path = "aligned_events.pkl"

    hr, gsr, motion, valence, arousal, panas = load_and_preprocess_data(data_path)
    print(f"Successfully loaded actual data.")
    print(f"Heart rate data shape: {hr.shape}")
    print(f"GSR data shape: {gsr.shape}")
    print(f"Motion data shape: {motion.shape}")
    print(f"Valence score: {valence.item()}")
    print(f"Arousal score: {arousal.item()}")
    print(f"PANAS scores: {panas}")
    
    # Initialize model
    feature_dim = 512  # Feature dimension for each modality
    model = MultiModalEmbedding(feature_dim=feature_dim)
    
    # Forward pass
    output, attention_weights = model(hr, gsr, motion, return_attention=True)
    
    # Print results
    print("\nModel output:")
    print(f"Output shape: {output.shape}")
    print(f"Output example (first 5 values): {output[0, 0, :5]}")
    
    # Visualize raw data, features and attention weights
    plt.figure(figsize=(15, 15))
    
    # Plot raw heart rate data
    plt.subplot(3, 2, 1)
    plt.plot(hr[0, 0, :100].cpu().numpy())
    plt.title('Raw Heart Rate Data (First 100 Samples)')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude')
    
    # Plot extracted heart rate features - select 3 feature channels to display
    plt.subplot(3, 2, 2)
    hr_features = model.hr_extractor(hr)
    for i in range(3):
        plt.plot(hr_features[0, i, :100].detach().cpu().numpy(), label=f'Feature {i+1}')
    plt.title('Heart Rate Features (First 3 Channels)')
    plt.xlabel('Time Point')
    plt.ylabel('Feature Value')
    plt.legend()
    
    # Plot raw GSR data
    plt.subplot(3, 2, 3)
    plt.plot(gsr[0, 0, :100].cpu().numpy())
    plt.title('Raw GSR Data (First 100 Samples)')
    plt.xlabel('Time Point')
    plt.ylabel('Amplitude')
    
    # Plot output features - select 3 feature channels to display
    plt.subplot(3, 2, 4)
    for i in range(3):
        plt.plot(output[0, :100, i].detach().cpu().numpy(), label=f'Fused Feature {i+1}')
    plt.title('Fused Features (First 3 Channels)')
    plt.xlabel('Time Point')
    plt.ylabel('Feature Value')
    plt.legend()
    
    # Visualize attention weights: Heart Rate attending to GSR
    plt.subplot(3, 2, 5)
    attn_hr_to_gsr = attention_weights['modality_0']['modality_1'][0].detach().cpu().numpy()
    plt.imshow(attn_hr_to_gsr[:50, :50], cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Attention: Heart Rate → GSR (First 50 Points)')
    plt.xlabel('GSR Time Points')
    plt.ylabel('HR Time Points')
    
    # Visualize attention weights: GSR attending to Motion
    plt.subplot(3, 2, 6)
    attn_gsr_to_motion = attention_weights['modality_1']['modality_2'][0].detach().cpu().numpy()
    plt.imshow(attn_gsr_to_motion[:50, :50], cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Attention: GSR → Motion (First 50 Points)')
    plt.xlabel('Motion Time Points')
    plt.ylabel('GSR Time Points')
    
    plt.tight_layout()
    plt.savefig('feature_extraction_test.png')
    plt.close()
    
    # Create a separate visualization to focus just on the parallel cross-attention weights
    plt.figure(figsize=(15, 10))
    
    attention_pairs = [
        ('HR → GSR', 'modality_0', 'modality_1'),
        ('HR → Motion', 'modality_0', 'modality_2'),
        ('GSR → HR', 'modality_1', 'modality_0'),
        ('GSR → Motion', 'modality_1', 'modality_2'),
        ('Motion → HR', 'modality_2', 'modality_0'),
        ('Motion → GSR', 'modality_2', 'modality_1')
    ]
    
    for i, (title, from_mod, to_mod) in enumerate(attention_pairs):
        plt.subplot(2, 3, i+1)
        attn = attention_weights[from_mod][to_mod][0].detach().cpu().numpy()
        plt.imshow(attn[:30, :30], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Attention: {title} (First 30 Points)')
        plt.xlabel(f'{to_mod.split("_")[1]} Time Points')
        plt.ylabel(f'{from_mod.split("_")[1]} Time Points')
    
    plt.tight_layout()
    plt.savefig('parallel_cross_attention.png')
    plt.close()
    
    print("\nTest completed! Feature extraction graph saved as 'feature_extraction_test.png'")
    print("Parallel cross-attention visualization saved as 'parallel_cross_attention.png'")
    print("Feature extractor now uses LayerNorm instead of BatchNorm, avoiding cross-subject information leakage!")
    print("Cross-attention mechanism has been added for better modality fusion!")
    print("Parallel Cross-Attention implemented: each modality attends to other modalities independently and combines the results!") 