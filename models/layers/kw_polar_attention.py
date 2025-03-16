import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from models.layers.bilinear_layer import Bilinear
from transformers.activations import gelu, gelu_new, silu
import sys

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
BertLayerNorm = torch.nn.LayerNorm
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": silu, "gelu_new": gelu_new, "mish": mish}

class KWBilinearAttention(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.hidden_dim = cus_config.attr_dim
        self.num_heads = cus_config.num_attr_heads

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Dimensionality per head
        self.head_dim = self.hidden_dim// self.num_heads 

        # Bilinear layers for Q, K, V
        self.WQ = Bilinear(self.hidden_dim, self.hidden_dim)  # For the first attention with n
        self.WK = Bilinear(self.hidden_dim, self.hidden_dim)
        self.WV = Bilinear(self.hidden_dim, self.hidden_dim)
        # Final linear projection
        self.final_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sequence, keyword, attention_mask):
        """
        Inputs:
        - sequence: Tensor of shape (batch_size, seq_length, hidden_dim)
        - keyword: Tensor of shape (1, 1, hidden_dim)

        Output:
        - context: Tensor of shape (batch_size, seq_length, hidden_dim)
        """
        batch_size, seq_length, _ = sequence.size()

        # Step 1: BiLinear transformations for Q, K, V
        Q = self.WQ(sequence,keyword)            # Shape: (batch_size, seq_length, hidden_dim)
        K = self.WK(sequence,keyword)          # Shape: (batch_size, seq_length, hidden_dim)
        V = self.WV(sequence,keyword)          # Shape: (batch_size, seq_length, hidden_dim)

        

        # Step 2: Split Q, K, V into multiple heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Step 3: Transpose for easier dot product
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        K = K.permute(0, 2, 1, 3)   # (batch_size, num_heads, seq_length, head_dim)
        V = V.permute(0, 2, 1, 3)   # (batch_size, num_heads, seq_length, head_dim)

        # Step 4: Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
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

class KWPolarattention(nn.Module):
    def __init__(self, config, cus_config):
        super(KWPolarattention, self).__init__()
        self.attentiontypepos = KWBilinearAttention(config, cus_config)
        self.attentiontypeneg = KWBilinearAttention(config, cus_config)
        self.kwcountpos = 50
        self.kwcountneg = 50
        self.batch_size = cus_config.TRAIN.batch_size 

        # Simple MLP (Multi-Layer Perceptron) for fusion
        self.mlppos = nn.Linear(self.kwcountpos, 1),  # Reduce the keyword dimension to 1
        self.mlpneg = nn.Linear(self.kwcountneg, 1),  # Reduce the keyword dimension to 1

    def forward(self, sequence, poskeywords, negkeywords, attention_mask):
        outputspos = []
        outputsneg = []
        

        # Apply attention for each keyword
        for i in range(self.kwcountpos):
            temppos = self.attentiontypepos(sequence, poskeywords[i], attention_mask)  # Shape: (batch_size, seq_len, hidden_dimension)
            outputspos.append(temppos)
        for i in range(self.kwcountneg):
            tempneg = self.attentiontypeneg(sequence, negkeywords[i], attention_mask)  # Shape: (batch_size, seq_len, hidden_dimension)
            outputsneg.append(tempneg)

        # Concatenate along the batch dimension
        stacked_output_pos = torch.stack(outputspos, dim=3)  # Shape: (batch_size, seq_len, hidden_dimension, kwcount)
        stacked_output_neg = torch.stack(outputsneg, dim=3) # Shape: (batch_size, seq_len, hidden_dimension, kwcount)
        
        # Apply MLP along the keyword dimension
        fused_output_pos = self.mlppos(stacked_output_pos).squeeze(-1)  # Shape: (batch_size, seq_len, hidden_dimension, 1(kwcount))
        fused_output_neg = self.mlpneg(stacked_output_neg).squeeze(-1)  # Shape: (batch_size, seq_len, hidden_dimension, 1)
        # Squeeze the keyword dimension (size 1) to get shape (batch_size, seq_len, hidden_dimension)
        output = torch.cat((fused_output_pos, fused_output_neg), dim=3)

        return output