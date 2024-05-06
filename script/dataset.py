from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
import torch
from processing import *

class TextClassificationDataset():
    def __init__(self, df, vocab, tokenizer):
        self.df = df
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.dataset = to_map_style_dataset(self.yeild_token())
        
    def yeild_token(self):
        for idx, row in self.df.iterrows():
            sentence = row['normalized_sentence']
            encoded_sentence = self.vocab(self.tokenizer(sentence)) 
            label = row['label']           
            yield encoded_sentence, label
    
    def get_dataloader(self, batch_size=128, shuffle=True, device=None):
        
        def collate_batch(batch):
            label_list, text_list, offset_list = [], [], [0]
            for encoded_sentence, label in batch:
                label_list.append(label)
                encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)
                text_list.append(encoded_sentence)
                offset_list.append(encoded_sentence.size(0))
            labels = torch.tensor(label_list, dtype=torch.int64)
            offset_list = torch.tensor(offset_list[:-1]).cumsum(dim=0)
            text = torch.cat(text_list)
            return text.to(device), labels.to(device), offset_list.to(device)
        
        return DataLoader(self.dataset, collate_fn=collate_batch, batch_size=batch_size, shuffle=shuffle)