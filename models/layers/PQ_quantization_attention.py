import torch
import torch.nn as nn
import numpy as np

class PQLinear(nn.Module):
    def __init__(self, in_features, out_features, num_subvectors, num_centroids):
        super(PQLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_subvectors = num_subvectors
        self.num_centroids = num_centroids

        # Split the weight matrix into subvectors
        assert in_features % num_subvectors == 0, "in_features must be divisible by num_subvectors"
        self.subvector_dim = in_features // num_subvectors

        # Initialize codebooks (centroids) for each subvector
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_centroids, self.subvector_dim))  # Codebook for each subvector
            for _ in range(num_subvectors)
        ])

        # Initialize assignments (indices to centroids) for each output feature and subvector
        self.assignments = nn.Parameter(torch.randint(0, num_centroids, (out_features, num_subvectors)))

    def forward(self, x):
        # Initialize the list for storing the quantized weight vectors for each subvector
        quantized_weights = []

        # Loop through each subvector
        for i in range(self.num_subvectors):
            # Get the indices from the assignments for this subvector
            indices = self.assignments[:, i]  # Shape: (out_features,)

            # Quantize the weights by selecting the centroids from the codebook
            quantized_weights.append(self.codebooks[i][indices])  # Shape: (out_features, subvector_dim)

        # Concatenate the quantized weights across all subvectors
        quantized_weights = torch.cat(quantized_weights, dim=1)  # Shape: (out_features, in_features)

        # Perform the linear transformation using the quantized weight matrix
        return torch.matmul(x, quantized_weights.T)  # Shape: (batch_size, out_features)


class PQAttention(nn.Module):
    def __init__(self, config, num_subvectors=4, num_centroids=256):
        super(PQAttention, self).__init__()
        self.config = config

        # Projection layers for query, key, and value
        self.q_proj = PQLinear(config.hidden_size, config.hidden_size // 2, num_subvectors, num_centroids)
        self.k1_proj = PQLinear(config.hidden_size, config.hidden_size // 2, num_subvectors, num_centroids)
        self.k2_proj = PQLinear(config.hidden_size, config.hidden_size // 2, num_subvectors, num_centroids)

        # Output fusion
        self.mlp = nn.Sequential(
            PQLinear(2 * config.hidden_size, config.hidden_size // 2, num_subvectors, num_centroids),
            nn.ReLU(),
            PQLinear(config.hidden_size // 2, config.hidden_size, num_subvectors, num_centroids)
        )

    def forward(self, Q, K1, K2):
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