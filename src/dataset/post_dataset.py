import logging

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from .randmask import get_test2_1_data, get_mask_token_ids

l2i = {
    "adhd": 0,
    "alcoholism": 1,
    "alzheimer": 2,
    "anger syndrome": 3,
    "anxiety": 4,
    "bipolar disorder": 5,
    "depression": 6,
    "eating disorder": 7,
    "insomnia": 8,
    "nicotine": 9,
    "schizophrenia": 10,
    "suicide": 11
}

class post_dataset(Dataset):
    def __init__(self, df):        
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):   
        text, label = self.data[idx]

        return {
            'text':text,
            'label':torch.tensor(l2i[label], dtype=torch.long), # type:Tensor
            'index':idx # type:Tensor
        }
