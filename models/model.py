import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BertModel, BertPreTrainedModel
from models.layers.multi_attr_transformer import MAALayer
from models.layers.classifier import BERTClassificationHead, BERTClassificationHeadWithAttribute
from models.layers.fusion_layer import Fusion
from transformers.pipelines import zero_shot_image_classification



class MAAModel(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.cus_config = kwargs['cus_config']
        self.type = self.cus_config.type # a,b,c,d
        #'usr_prd', 'usr_ctgy', 'prd_ctgy', 'usr_prd_ctgy'
        
        if(self.cus_config.attributes == 'usr_ctgy'):
            # User embedding
            self.usr_embed = nn.Embedding(self.cus_config.num_usrs, self.cus_config.attr_dim)
            self.usr_embed.weight.requires_grad = True
            self.usr_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_usrs, self.cus_config.attr_dim) * 0.5 - 0.25)
            # Category embedding
            self.ctgy_embed = nn.Embedding(self.cus_config.num_ctgy, self.cus_config.attr_dim)
            self.ctgy_embed.weight.requires_grad = True
            self.ctgy_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_ctgy, self.cus_config.attr_dim) * 0.5 - 0.25)
        elif(self.cus_config.attributes == 'prd_ctgy'):
            # Product embedding
            self.prd_embed = nn.Embedding(self.cus_config.num_prds, self.cus_config.attr_dim)
            self.prd_embed.weight.requires_grad = True
            self.prd_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_prds, self.cus_config.attr_dim) * 0.5 - 0.25)
            # Category embedding
            self.ctgy_embed = nn.Embedding(self.cus_config.num_ctgy, self.cus_config.attr_dim)
            self.ctgy_embed.weight.requires_grad = True
            self.ctgy_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_ctgy, self.cus_config.attr_dim) * 0.5 - 0.25)
        elif(self.cus_config.attributes == 'usr_prd_ctgy'):
            # User embedding
            self.usr_embed = nn.Embedding(self.cus_config.num_usrs, self.cus_config.attr_dim)
            self.usr_embed.weight.requires_grad = True
            self.usr_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_usrs, self.cus_config.attr_dim) * 0.5 - 0.25)
            # Product embedding
            self.prd_embed = nn.Embedding(self.cus_config.num_prds, self.cus_config.attr_dim)
            self.prd_embed.weight.requires_grad = True
            self.prd_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_prds, self.cus_config.attr_dim) * 0.5 - 0.25)
            # Category embedding
            self.ctgy_embed = nn.Embedding(self.cus_config.num_ctgy, self.cus_config.attr_dim)
            self.ctgy_embed.weight.requires_grad = True
            self.ctgy_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_ctgy, self.cus_config.attr_dim) * 0.5 - 0.25)
        else:
            self.usr_embed = nn.Embedding(self.cus_config.num_usrs, self.cus_config.attr_dim)
            self.usr_embed.weight.requires_grad = True
            self.usr_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_usrs, self.cus_config.attr_dim) * 0.5 - 0.25)
            # Product embedding
            self.prd_embed = nn.Embedding(self.cus_config.num_prds, self.cus_config.attr_dim)
            self.prd_embed.weight.requires_grad = True
            self.prd_embed.weight = nn.Parameter(torch.rand(self.cus_config.num_prds, self.cus_config.attr_dim) * 0.5 - 0.25)        
        

        if self.type not in ['b', 'a']:
          self.text = nn.Parameter(torch.rand(1, self.cus_config.attr_dim) * 0.5 - 0.25)
          # print(f"After uniform initialization: {self.text}")
                       
          self.ATrans_decoder = nn.ModuleList([MAALayer(config, self.cus_config) for _ in range(self.cus_config.n_mmalayer)])
          self.classifier = BERTClassificationHead(config)
        elif self.type == 'a':
            self.fusion = Fusion(self.config.hidden_size,self.cus_config.attr_dim)
            self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.cus_config.n_mmalayer)])
            self.classifier = BERTClassificationHead(config)
        else:
            self.classifier = BERTClassificationHeadWithAttribute(self.cus_config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attrs=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True,
            output_hidden_states=True
        )
        last_output = outputs[0] # last hidden_state in BERT
        pool_output = outputs[1] # pooled hidden_state over [CLS] in the last layer
        all_hidden_states, all_attentions = outputs[2:] # (bs,)

        usrs, prds, ctgys = attrs # (bs, ) * 2(3??)
        
            
        if(self.cus_config.attributes == 'usr_ctgy'):
            usr = self.usr_embed(usrs) # (bs, attr_dim)
            ctgy = self.ctgy_embed(ctgys) # (bs, attr_dim)
            if self.type == 'b':
                hidden_state = self.dropout(last_output)
                outputs = self.classifier(hidden_state, [usr, ctgy])
            elif self.type == 'a':
                extend_attention_mask = self.get_attention_mask(attention_mask)
                if 12 > self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                hidden_state = self.fusion(last_output, [usr, ctgy])
                hidden_state = self.dropout(hidden_state)
                for i, l in enumerate(self.layer):
                    hidden_state = l(hidden_state, extend_attention_mask)[0]
                hidden_state = self.dropout(hidden_state)
                outputs = self.classifier(hidden_state)
            else:
                
                t_self = self.text.expand_as(usr)  # (bs, attr_dim)
            
                extend_attention_mask = self.get_attention_mask(attention_mask)
            
                if 12 >= self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                    
                    hidden_state = self.dropout(last_output)
                    
                    for i, mmalayer in enumerate(self.ATrans_decoder):
                        hidden_state = mmalayer([usr, ctgy, t_self], hidden_state, extend_attention_mask)
                    
                    hidden_state = self.dropout(hidden_state)
                    
                    outputs = self.classifier(hidden_state)
                

            return (outputs, hidden_state)  # logits, last_hidden_stat
            
        elif(self.cus_config.attributes == 'prd_ctgy'):
            
            prd = self.prd_embed(prds) # (bs, attr_dim)
            ctgy = self.ctgy_embed(ctgys) # (bs, attr_dim)
            
            if self.type == 'b':
                hidden_state = self.dropout(last_output)
                outputs = self.classifier(hidden_state, [prd, ctgy])
            elif self.type == 'a':
                extend_attention_mask = self.get_attention_mask(attention_mask)
                if 12 > self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                hidden_state = self.fusion(last_output, [prd, ctgy])
                hidden_state = self.dropout(hidden_state)
                for i, l in enumerate(self.layer):
                    hidden_state = l(hidden_state, extend_attention_mask)[0]
                hidden_state = self.dropout(hidden_state)
                outputs = self.classifier(hidden_state)
            else:
                
                t_self = self.text.expand_as(prd)  # (bs, attr_dim)
            
                extend_attention_mask = self.get_attention_mask(attention_mask)
            
                if 12 >= self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                    
                    hidden_state = self.dropout(last_output)
                    
                    for i, mmalayer in enumerate(self.ATrans_decoder):
                        hidden_state = mmalayer([prd, ctgy, t_self], hidden_state, extend_attention_mask)
                    
                    hidden_state = self.dropout(hidden_state)
                    
                    outputs = self.classifier(hidden_state)
                

            return (outputs, hidden_state)  # logits, last_hidden_stat
        elif(self.cus_config.attributes == 'usr_prd_ctgy'):
            usr = self.usr_embed(usrs) # (bs, attr_dim)
            prd = self.prd_embed(prds) # (bs, attr_dim)
            ctgy = self.ctgy_embed(ctgys) # (bs, attr_dim)
            if self.type == 'b':
                hidden_state = self.dropout(last_output)
                outputs = self.classifier(hidden_state, [usr, prd, ctgy])
            elif self.type == 'a':
                extend_attention_mask = self.get_attention_mask(attention_mask)
                if 12 > self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                hidden_state = self.fusion(last_output, [usr, prd, ctgy])
                hidden_state = self.dropout(hidden_state)
                for i, l in enumerate(self.layer):
                    hidden_state = l(hidden_state, extend_attention_mask)[0]
                hidden_state = self.dropout(hidden_state)
                outputs = self.classifier(hidden_state)
            else:
                
                t_self = self.text.expand_as(usr)  # (bs, attr_dim)
            
                extend_attention_mask = self.get_attention_mask(attention_mask)
            
                if 12 >= self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                    
                    hidden_state = self.dropout(last_output)
                    
                    for i, mmalayer in enumerate(self.ATrans_decoder):
                        hidden_state = mmalayer([usr, prd, ctgy, t_self], hidden_state, extend_attention_mask)
                    
                    hidden_state = self.dropout(hidden_state)
                    
                    outputs = self.classifier(hidden_state)
                

            return (outputs, hidden_state)  # logits, last_hidden_stat
        else:
            usr = self.usr_embed(usrs) # (bs, attr_dim)
            prd = self.prd_embed(prds) # (bs, attr_dim)
            if self.type == 'b':
                hidden_state = self.dropout(last_output)
                outputs = self.classifier(hidden_state, [usr, prd])
            elif self.type == 'a':
                extend_attention_mask = self.get_attention_mask(attention_mask)
                if 12 > self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                hidden_state = self.fusion(last_output, [usr, prd])
                hidden_state = self.dropout(hidden_state)
                for i, l in enumerate(self.layer):
                    hidden_state = l(hidden_state, extend_attention_mask)[0]
                hidden_state = self.dropout(hidden_state)
                outputs = self.classifier(hidden_state)
            else:
                
                t_self = self.text.expand_as(usr)  # (bs, attr_dim)
            
                extend_attention_mask = self.get_attention_mask(attention_mask)
            
                if 12 >= self.cus_config.n_bertlayer > 0:
                    last_output = all_hidden_states[-(self.config.num_hidden_layers + 1 -self.cus_config.n_bertlayer)]
                    
                    hidden_state = self.dropout(last_output)
                    
                    for i, mmalayer in enumerate(self.ATrans_decoder):
                        hidden_state = mmalayer([usr, prd, t_self], hidden_state, extend_attention_mask)
                    
                    hidden_state = self.dropout(hidden_state)
                    
                    outputs = self.classifier(hidden_state)
                

            return (outputs, hidden_state)  # logits, last_hidden_stat
            
        
        

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids or attention_mask"
                )
        try:
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        except:
            print(extended_attention_mask)
            exit()
        # extended_attention_mask = ~extended_attention_mask * -10000.0
        return extended_attention_mask
    def log_tensor(self, tensor, name):
      if torch.isnan(tensor).any():
          print(f"\nNaN detected in {name}")
      elif torch.isinf(tensor).any():
          print(f"\nInf detected in {name}")
      else:
          print(f"\n{name} output is valid with mean: {tensor.mean().item()}")