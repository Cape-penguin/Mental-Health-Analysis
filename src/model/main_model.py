import logging
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import BertForMaskedLM, BertConfig

class MyModel(nn.Module):
    def __init__(self, tokenizer, config, device):
        super().__init__()

        self.tokenizer = tokenizer 
        self.device = device
        num_classes = config['num_classes']
        
        self.text_encoder = BertForMaskedLM.from_pretrained(
            config['text_encoder'], config=BertConfig.from_json_file(config['bert_config'])
        )
        
        embed_dim = config['embed_dim'] # 256
        text_width = self.text_encoder.config.hidden_size # 768
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.mlm_probability = 0.15
        
        self.mlp_head=nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0), # for simplicity
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, text):
        
        ##================= MLP head ===================##
        text_output=self.text_encoder.bert(text[0].input_ids, attention_mask = text[0].attention_mask,                      
                                            return_dict = True, )            
        text_embeds=text_output.last_hidden_state
        text_feat_i=F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        logits=self.mlp_head(text_feat_i)
        
        ##================= ssl ========================##
        text_output=self.text_encoder.bert(text[1].input_ids, attention_mask = text[1].attention_mask,                      
                                            return_dict = True, )            
        text_embeds=text_output.last_hidden_state
        text_feat_j=F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        
        ##================= MLM ========================## 
        input_ids = text[1].input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, device=self.device, targets=labels,
                                      probability_matrix = probability_matrix) 
        
        mlm_output = self.text_encoder(input_ids, 
                                       attention_mask = text[1].attention_mask,   
                                       return_dict = True,
                                       labels = labels,   
                                      )                           
        loss_mlm = mlm_output.loss

        return logits, text_feat_i, text_feat_j, loss_mlm

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):
        return super().__str__()