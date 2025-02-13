import torch
import torch.nn as nn

class LightweightAttention(nn.Module):
    def __init__(self, config):
        super(LightweightAttention, self).__init__()
        self.config = config
        
        # Projection layers for query, key, and value
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False)
        self.k1_proj = nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False)
        self.k2_proj = nn.Linear(config.hidden_size, config.hidden_size // 2, bias=False)
        
        # Output fusion
        self.mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )


    def forward(self, Q, K1, K2):
        """
        Q: Query tensor [batch_size, seq_len, hidden_size]
        K1: Key tensor 1 [n, hidden_size]
        K2: Key tensor 2 [m, hidden_size]
        """
        # Ensure everything is on the same device
        device = Q.device
        K1 = K1.to(device)
        K2 = K2.to(device)

        # Project queries and keys to lower-dimensional space
        Q_proj = self.q_proj(Q)  # Shape: [batch_size, seq_len, hidden_size//2]
        K1_proj = self.k1_proj(K1)  # Shape: [n, hidden_size//2]
        K2_proj = self.k2_proj(K2)  # Shape: [m, hidden_size//2]

        # Compute attention scores
        attn_scores1 = torch.matmul(Q_proj, K1_proj.T) / (self.config.hidden_size ** 0.5)  # Shape: [batch_size, seq_len, n]
        attn_scores2 = torch.matmul(Q_proj, K2_proj.T) / (self.config.hidden_size ** 0.5)  # Shape: [batch_size, seq_len, m]

        # Apply softmax to get attention weights
        attn_weights1 = torch.softmax(attn_scores1, dim=-1)  # Shape: [batch_size, seq_len, n]
        attn_weights2 = torch.softmax(attn_scores2, dim=-1)  # Shape: [batch_size, seq_len, m]

        # Compute attended outputs
        attn_output1 = torch.matmul(attn_weights1, K1)  # Shape: [batch_size, seq_len, hidden_size]
        attn_output2 = torch.matmul(attn_weights2, K2)  # Shape: [batch_size, seq_len, hidden_size]

        # Concatenate outputs and apply MLP fusion
        fused_output = torch.cat([attn_output1, attn_output2], dim=-1)  # Shape: [batch_size, seq_len, 2 * hidden_size]
        output = self.mlp(fused_output)  # Shape: [batch_size, seq_len, hidden_size]

        return output
