import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(
        self,
        max_vocab=10004,
        emb_size=100,
        filter=400,
        kernel=3
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_vocab, emb_size) # (10000, 100)
        self.conv1d = nn.Conv1d(emb_size, filter, kernel)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(filter, 1)
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, x, true):
        '''
        :param x: shape (batch, max_len)
        '''
        x = self.embedding(x).transpose(1, 2) # (batch, emb_size, max_len)
        x = F.relu(self.conv1d(x)) # (batch, filter_size, max_len-2)
        x = self.max_pool(x).squeeze(-1) # (batch, filter_size, 1)
        x = torch.sigmoid(self.fc(x).squeeze(-1))
        if true is not None:
            loss = self.bceloss(x, true.float())
            return loss, x
        else:
            return x