"""
Author: Etlazure
Creation Date: March 3, 2025
Purpose: Implement a Transformer-based encoder for multi-modal emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in the Transformer paper.
    """
    def __init__(self, d_model, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        d_k = k.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        
        # Merge heads
        context = self.merge_heads(context)
        
        # Final linear layer
        output = self.wo(context)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """
    Feed-forward network with residual connection
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """
    Encoder layer with multi-head self-attention and feed-forward network with residual connection
    """
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.2):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Only need one LayerNorm for feed-forward's residual connection
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head self-attention (no residual connection)
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        
        # Feed-forward network with residual connection
        ff_output = self.feed_forward(attn_output)
        ff_output = self.dropout2(ff_output)
        out = self.norm(attn_output + ff_output)  # Residual connection
        
        return out, attn_weights

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for multi-modal feature fusion.
    Takes pre-encoded features from feature_extraction.py and applies N layers of 
    self-attention and feed-forward processing.
    """
    def __init__(self, d_model=1536, num_heads=8, num_layers=4, d_ff=2048, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Global average pooling will be applied in forward pass
        
    def forward(self, x, mask=None):
        """
        Forward pass through N encoder layers
        
        Parameters:
            x: Input tensor with shape [batch_size, seq_len, d_model]
               This should be the concatenated and position-encoded features 
               from feature_extraction.py
            mask: Optional mask tensor
            
        Returns:
            output: Tensor of shape [batch_size, d_model] after global average pooling
            attention_weights: List of attention weights from each layer
        """
        x = self.dropout(x)
        
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Global average pooling over the sequence length dimension
        output = torch.mean(x, dim=1)  # Shape: [batch_size, d_model]
        
        return output, attention_weights

def visualize_attention(attention_weights, layer_idx=0, head_idx=0, save_path=None):
    """
    Visualize attention weights from a specific layer and head
    """
    plt.figure(figsize=(10, 8))
    
    attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
    
    plt.imshow(attn[:50, :50], cmap='viridis')
    plt.colorbar()
    plt.title(f'Layer {layer_idx}, Head {head_idx} Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Test code
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create synthetic data for testing
    batch_size = 4
    seq_len = 100
    d_model = 1536  # 3 * 512 (concatenated features from 3 modalities)
    
    # Create random input tensor (simulating concatenated and position-encoded features)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Define model parameters
    num_heads = 8
    num_layers = 4
    d_ff = 2048
    dropout = 0.2
    
    # Initialize model
    model = TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # Forward pass
    encoded_features, attention_weights = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Encoded features shape: {encoded_features.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")
    print(f"Attention weights shape: {attention_weights[0].shape}")
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Visualize attention weights
    visualize_attention(attention_weights, layer_idx=0, head_idx=0, save_path="self_attention_visualization.png")
    print("Attention visualization saved to 'self_attention_visualization.png'")
    
    # Test with different batch sizes
    batch_sizes = [1, 8, 16]
    for bs in batch_sizes:
        test_input = torch.randn(bs, seq_len, d_model)
        out, _ = model(test_input)
        print(f"Output shape for batch size {bs}: {out.shape}")
    
    print("Test completed successfully!") 