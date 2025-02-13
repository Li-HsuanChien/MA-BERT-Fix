import torch
import torch.nn as nn

class BilinearAttention(nn.Module):
    def __init__(self, config):
        super(BilinearAttention, self).__init__()
        self.config = config
        
        
        # Bilinear weight matrices
        self.W1 = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))  # For the first attention with n
        self.W2 = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))  # For the second attention with m
        
        # MLP for fusion
        self.mlp = nn.Sequential(
            nn.Linear(2 * config.hidden_size, config.hidden_size),  # Concatenated output of 2 bilinears
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)  # Output size after MLP
        )

    def forward(self, Q, K1, K2):
        """
        Q: Query tensor [batch_size, seq_len, config.hidden_size]
        K1: Key1 tensor [n, config.hidden_size]
        K2: Key2 tensor [m, config.hidden_size]
        """
        device = Q.device
        self.W1 = self.W1.to(device)  # Ensure W1 is on the same device
        self.W2 = self.W2.to(device)  # Ensure W2 is on the same device
        K1 = K1.to(device)  # Ensure K1 is on the same device
        K2 = K2.to(device)  # Ensure K2 is on the same device
        
        
        # Derive n and m from the shapes of K1 and K2
        n = K1.shape[0]  # First dimension of K1
        m = K2.shape[0]  # First dimension of K2
        
        # Step 1: Bilinear attention with K1 (n)
        
        K1 = K1.view(n, self.config.hidden_size)  # Key1 tensor with shape [n, embed_dim]
        K2 = K2.view(m, self.config.hidden_size)  # Key2 tensor with shape [m, embed_dim]
        Q1 = Q
        Q2 = Q
        transformed1 = torch.matmul(Q1, self.W1) # Shape: (batch_size, seq_len, embed_dim)
        transformed2 = torch.matmul(Q2, self.W2) # Shape: (batch_size, seq_len, embed_dim)
        
        
        
        attention_score1 = torch.matmul(transformed1, K1.T) # Shape: (batch_size, seq_len, n)
        attention_score2 = torch.matmul(transformed2, K2.T) # Shape: (batch_size, seq_len, m)
        
        # Bilinear attention with Key1 (Q * W1 * K1.T)
        attention_weights1 = torch.softmax(attention_score1,dim=-1)  # Shape: (batch_size, seq_len, n)
        # Bilinear attention with Key2 (Q * W2 * K2.T)
        attention_weights2 = torch.softmax(attention_score2,dim=-1)  # Shape: (batch_size, seq_len, m)
        
        bilinear_out1 = torch.matmul(attention_weights1, K1)  # [batch_size, seq_len, embed_dim]
        bilinear_out2 = torch.matmul(attention_weights2, K2)  # [batch_size, seq_len, embed_dim]
        
        fused_out = torch.cat([bilinear_out1, bilinear_out2], dim=-1)  # Shape: [batch_size, seq_len, 2*(embed_dim)]
        
        fused_out = self.mlp(fused_out)  # Shape: [batch_size, seq_len, embed_dim]
        
        return fused_out