import torch
import torch.nn as nn
import torch.nn.functional as F

class KWMultiHeadAttention(nn.Module):
    def __init__(self, config, cus_config):
        super(KWMultiHeadAttention, self).__init__()
        self.hidden_dim = cus_config.attr_dim
        self.num_heads = cus_config.num_attr_heads

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads  == 0, "hidden_dim must be divisible by num_heads"

        # Dimensionality per head
        self.head_dim = self.hidden_dim// self.num_heads 
        # Linear layers for Q, K, V
        self.W_Q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_K = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_V = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Final linear projection
        self.final_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sequence, keyword):
        """
        Inputs:
        - sequence: Tensor of shape (batch_size, seq_length, hidden_dim)
        - keyword: Tensor of shape (1, 1, hidden_dim)

        Output:
        - context: Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        batch_size, seq_length, _ = sequence.size()

        # Step 1: Linear transformations for Q, K, V
        Q = self.W_Q(sequence)            # Shape: (batch_size, seq_length, hidden_dim)
        K = self.W_K(keyword)             # Shape: (1, 1, hidden_dim)
        V = self.W_V(keyword)             # Shape: (1, 1, hidden_dim)

        # Step 2: Split Q, K, V into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(1, 1, self.num_heads, self.head_dim)
        V = V.view(1, 1, self.num_heads, self.head_dim)

        # Step 3: Transpose for easier dot product
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, 1, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, 1, head_dim)

        # Step 4: Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Step 5: Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=-2)  # Shape: (batch_size, num_heads, seq_length, 1)

        # Step 6: Apply attention weights to V
        context = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        # Step 7: Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_length, num_heads, head_dim)
        context = context.view(batch_size, seq_length, self.hidden_dim)  # Shape: (batch_size, seq_length, hidden_dim)

        # Step 8: Final linear projection
        output = self.final_projection(context)  # Shape: (batch_size, seq_length, hidden_dim)

        return output


class KWBilinearAttention(nn.Module):
    def __init__(self, config, cus_config):
        super(KWBilinearAttention, self).__init__()
        self.hidden_dim = cus_config.attr_dim
        self.num_heads = cus_config.num_attr_heads

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Dimensionality per head
        self.head_dim = self.hidden_dim// self.num_heads 

        # Bilinear layers for Q, K, V
        self.WQ = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))  # For the first attention with n
        self.WK = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.WV = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # Final linear projection
        self.final_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sequence, keyword):
        """
        Inputs:
        - sequence: Tensor of shape (batch_size, seq_length, hidden_dim)
        - keyword: Tensor of shape (1, 1, hidden_dim)

        Output:
        - context: Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        batch_size, seq_length, _ = sequence.size()

        # Step 1: Linear transformations for Q, K, V
        Q = torch.matmul(sequence,self.WQ) * keyword            # Shape: (batch_size, seq_length, hidden_dim)
        K = torch.matmul(sequence,self.WK) * keyword             # Shape: (1, 1, hidden_dim)
        V = torch.matmul(sequence,self.WV) * keyword             # Shape: (1, 1, hidden_dim)

        print((Q.size()))
        print((K.size()))
        print((V.size()))

        

        # Step 2: Split Q, K, V into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Step 3: Transpose for easier dot product
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, 1, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, 1, head_dim)

        # Step 4: Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Step 5: Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=-2)  # Shape: (batch_size, num_heads, seq_length, 1)

        # Step 6: Apply attention weights to V
        context = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, seq_length, head_dim)

        # Step 7: Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, seq_length, num_heads, head_dim)
        context = context.view(batch_size, seq_length, self.hidden_dim)  # Shape: (batch_size, seq_length, hidden_dim)

        # Step 8: Final linear projection
        output = self.final_projection(context)  # Shape: (batch_size, seq_length, hidden_dim)

        return output


class KWattention(nn.Module):
    def __init__(self, config, cus_config):
        super(KWattention, self).__init__()
        self.attentiontype = KWMultiHeadAttention(config, cus_config)
        self.kwcount = cus_config.num_kws

        # Simple MLP (Multi-Layer Perceptron) for fusion
        self.mlp = nn.Sequential(
            nn.Linear(self.kwcount, config.hidden_size),  # First Linear Layer
            nn.ReLU(),                                          # Activation
            nn.Linear(config.hidden_size, 1)   # Second Linear Layer
        )

    def forward(self, sequence, keywords):
        outputs = []

        # Apply attention for each keyword
        for i in range(self.kwcount):
            temp = self.attentiontype(sequence, keywords[i].unsqueeze(0))  # Shape: (batch_size, seq_len, hidden_dimension)
            outputs.append(temp)

        # Concatenate along the batch dimension
        concatenated_output = torch.cat(outputs, dim=0)  # Shape:(batch_size * kwcount, seq_len, hidden_dimension)

        # Apply MLP to fuse the features
        fused_output = torch.mean(concatenated_output, dim=0, keepdim=True)  # Shape: (1, seq_len, hidden_dimension)

        return fused_output



