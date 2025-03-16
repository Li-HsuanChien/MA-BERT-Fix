import torch

class ContextualKeywordBERT(torch.nn.Module):

    def __init__(self, cus_config):

        super(ContextualKeywordBERT, self).__init__()


        self.cross_attention1 = torch.nn.MultiheadAttention(embed_dim=cus_config.attr_dim, num_heads=cus_config.num_heads)

        self.compress_keywords = torch.nn.Linear(cus_config.attr_dim, cus_config.attr_dim)  # Linear layer for compression

        self.cross_attention2 = torch.nn.MultiheadAttention(embed_dim=cus_config.attr_dim, num_heads=cus_config.num_heads)

 

    def forward(self, seq_hidden_state, attention_mask, keyword_embeddins):

 

        # Initial cross-attention: keywords as queries, BERT output as keys and values

        enriched_keywords, _ = self.cross_attention1(keyword_embeddins, seq_hidden_state, seq_hidden_state, attn_mask=attention_mask)

 

        # Compress keyword information using a linear layer

        compressed_keywords = self.compress_keywords(enriched_keywords)

 

        # Extract [CLS] token representation

        cls_token = seq_hidden_state[:, 0, :].unsqueeze(1)  # Shape: [batch size, 1, embedding size]

 

        # Second cross-attention: [CLS] token as query, compressed keywords as keys and values

        cls_enriched, _ = self.cross_attention2(cls_token, compressed_keywords, compressed_keywords)

 

        # Combine [CLS] token and enriched [CLS] token information

        combined_output = torch.cat((cls_token.squeeze(1), cls_enriched.squeeze(1)), dim=-1)

        return combined_output