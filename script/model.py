import torch.nn as nn
import torch

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode='sum')
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.weight_initilization()
        
    def weight_initilization(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc(embedded)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def predict(self, text, offsets):
        with torch.no_grad():
            output = self.forward(text, offsets)
            prob = torch.softmax(output, dim=1)
            label = output.argmax(1)
            return prob, label