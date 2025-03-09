import torch
from torch.cuda.amp.autocast_mode import custom_fwd
import torch.nn as nn
import torch.nn.functional as F
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


class KWattentionLayer(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.attentiontype = KWBilinearAttention(config, cus_config)
        self.hidden_dim = cus_config.attr_dim
        self.poskwcount = cus_config.num_posembed
        self.negkwcount = cus_config.num_negembed
        # self.kwcount = cus_config.kw_attention_nums
        self.kwcount = 100
        self.batch_size = cus_config.TRAIN.batch_size 
        self.pad_tensor = torch.zeros((abs(self.poskwcount - self.negkwcount), self.hidden_dim)).to(cus_config.device)  # Padding with zeros

        # Simple MLP (Multi-Layer Perceptron) for fusion
        self.mlp = nn.Sequential(
            nn.Linear(self.kwcount, 1),  # Reduce the keyword dimension to 1
        )

    def forward(self, hidden_state, positive_keywords, negative_keywords, attention_mask):
        outputs = []
        # Padding the smaller tensor
        diff = self.poskwcount - self.negkwcount
        if diff > 0:
            negative_keywords = torch.cat((negative_keywords, self.pad_tensor), dim=0)
        elif diff < 0:
            positive_keywords = torch.cat((positive_keywords, self.pad_tensor), dim=0)
        keyword_pool = torch.stack((positive_keywords, negative_keywords), dim=1).reshape(-1, positive_keywords.shape[1])
        # # Interleave along dim=0
       
        # keyword_pool = torch.cat((positive_keywords, negative_keywords), dim=0)
        # Apply attention for each keyword
        for i in range(self.kwcount):
            temp = self.attentiontype(hidden_state, keyword_pool[i].unsqueeze(0), attention_mask)  # Shape: (batch_size, seq_len, hidden_dimension)
            outputs.append(temp)

        # Concatenate along the batch dimension
        stacked_output = torch.stack(outputs, dim=3)  # Shape: (batch_size, seq_len, hidden_dimension, kwcount)

        # Apply MLP along the keyword dimension
        fused_output = self.mlp(stacked_output)  # Shape: (batch_size, seq_len, hidden_dimension, 1)

        # Squeeze the keyword dimension (size 1) to get shape (batch_size, seq_len, hidden_dimension)
        fused_output = fused_output.squeeze(-1)

        return fused_output

class KWattention(nn.Module):
    def __init__(self, config, cus_config):
        super().__init__()
        self.attentionLayer = KWattentionLayer(config, cus_config)
        self.bottleneck = Bottleneck(cus_config)

    def forward(self, hidden_state, positive_keywords, negative_keywords, attention_mask):
        attention_output = self.attentionLayer(hidden_state, positive_keywords, negative_keywords, attention_mask)
        outputs = self.bottleneck(attention_output)
        return outputs

class Bottleneck(nn.Module):
    def __init__(self, cus_config):
        super().__init__()
        self.densedown = nn.Linear(cus_config.attr_dim, cus_config.intermediate_size)
        if isinstance(cus_config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[cus_config.hidden_act]
        else:
            self.intermediate_act_fn = cus_config.hidden_act

        self.denseup = nn.Linear(cus_config.intermediate_size, cus_config.attr_dim)
        self.LayerNorm = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)
        self.dropout = nn.Dropout(cus_config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states_down = self.densedown(hidden_states)
        hidden_states_down = self.intermediate_act_fn(hidden_states_down)
        hidden_states_up = self.denseup(hidden_states_down)


        hidden_states_up = self.dropout(hidden_states_up)
        hidden_states = self.LayerNorm(hidden_states_up + hidden_states)

        return hidden_states


