import torch
import torch.nn as nn
import torch.nn.functional as F
BertLayerNorm = torch.nn.LayerNorm
from transformers.activations import gelu
class KWMultiheadAttention(nn.Module):
    def __init__(self, cus_config):
        super().__init__()
        self.hidden_dim = cus_config.attr_dim
        self.num_heads = cus_config.num_attr_heads
        
        self.query = nn.Linear(cus_config.attr_dim, cus_config.attr_dim)
        self.key = nn.Linear(cus_config.attr_dim, cus_config.attr_dim)
        self.value = nn.Linear(cus_config.attr_dim, cus_config.attr_dim)

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Dimensionality per head
        self.head_dim = self.hidden_dim// self.num_heads 

        self.final_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, KV, Q, attention_mask=None):
        """
        Inputs:
        - KV: Tensor of shape (batch_size, KV_size, hidden_dim)
        - keyword: Tensor of shape (batch_size, Q_Size, hidden_dim)

        Output:
        - context: Tensor of shape (batch_size, KV_size, hidden_dim)
        """
        batch_size, KV_size, _ = KV.size()
        batch_size, Q_Size, _ = Q.size()

        Q = self.query(Q)
        K = self.key(KV)       
        V = self.key(KV)       

        
        Q = Q.view(batch_size, Q_Size, self.num_heads, self.head_dim)
        K = K.view(batch_size, KV_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, KV_size, self.num_heads, self.head_dim)
        # Step 3: Transpose for easier dot product
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, Q_Size, head_dim)
        K = K.permute(0, 2, 1, 3)   # (batch_size, num_heads, KV_size, head_dim)
        V = V.permute(0, 2, 1, 3)   # (batch_size, num_heads, KV_size, head_dim)

        # Step 4: Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Step 5: Softmax normalization
        attention_weights = F.softmax(attention_scores, dim=-2)  # Shape: (batch_size, num_heads, KV_size, 1)

        # Step 6: Apply attention weights to V
        context = torch.matmul(attention_weights, V)  # Shape: (batch_size, num_heads, Q_size, head_dim)
        # Step 7: Concatenate heads
        context = context.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, Q_Size, num_heads, head_dim)
        context = context.view(batch_size, Q_Size, self.hidden_dim)  # Shape: (batch_size, Q_Size, hidden_dim)

        # Step 8: Final linear projection
        output = self.final_projection(context)  # Shape: (batch_size, Q_Size, hidden_dim)

        return output


class ContextualKeywordBERT(nn.Module):

    def __init__(self, cus_config):
        super(ContextualKeywordBERT, self).__init__()

        self.cus_config = cus_config
        self.cross_attention1 = KWMultiheadAttention(self.cus_config)
        self.cross_attention2 = KWMultiheadAttention(self.cus_config)

       

        # LayerNorms
        self.norm1 = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)
        self.norm2 = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)
        self.norm3 = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)
        self.norm4 = BertLayerNorm(cus_config.attr_dim, eps=cus_config.layer_norm_eps)

        # Intermediate projection layers
        self.intermediate1 = nn.Linear(cus_config.attr_dim, cus_config.intermediate_size) 
        self.intermediate2 = nn.Linear(cus_config.attr_dim, cus_config.intermediate_size)  

        self.output1 = nn.Linear(cus_config.intermediate_size, cus_config.attr_dim)  
        self.output2 = nn.Linear(cus_config.intermediate_size, cus_config.attr_dim)  
        


    def forward(self, seq_hidden_state, keyword_embeddings, attention_mask):
        
        keyword_embeddings = keyword_embeddings.unsqueeze(0).expand(seq_hidden_state.size(dim=0), -1, -1)
        
        extended_attention_mask = attention_mask.expand(-1, self.cus_config.num_attr_heads, keyword_embeddings.size(dim=1), -1)

        # --- First cross-attention block ---
        enriched_keywords = self.cross_attention1(seq_hidden_state, keyword_embeddings, None)
        midway1 = enriched_keywords + keyword_embeddings  # Add residual connection
        midway1 = self.norm1(midway1)  # LayerNorm

        # Feed-forward layer
        hidden1 = self.intermediate1(midway1)  # TODO: Edit activation function
        hidden1 = gelu(hidden1)  # Placeholder activation
        hidden1 = self.output1(hidden1)
        enriched_keywords = self.norm2(midway1 + hidden1)  # Add residual and normalize


        # --- Second cross-attention block ---
        cls_token = seq_hidden_state[:, 0, :].unsqueeze(1)  
        cls_enriched = self.cross_attention2(enriched_keywords, cls_token)

        midway2 = cls_enriched + cls_token  # Add residual connection
        midway2 = self.norm3(midway2)  # LayerNorm

        # Feed-forward layer
        hidden2 = self.intermediate2(midway2)  # TODO: Edit activation function
        hidden2 = gelu(hidden2)  # Placeholder activation
        hidden2 = self.output2(hidden2)
        cls_enriched = self.norm4(midway2 + hidden2)  # Add residual and normalize

        combined_output = torch.cat((cls_token.squeeze(1), cls_enriched.squeeze(1)), dim=-1)
        return combined_output