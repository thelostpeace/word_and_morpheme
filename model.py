import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class Config():
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
            for key in kwargs:
                setattr(self, key, kwargs[key])

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False, indent=4)

    def set(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
            for key in kwargs:
                setattr(self, key, kwargs[key])

class TextCNN(nn.Module):
    """
        TextCNN model
    """
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size                     # vocab size
        self.emb_dim = config.embedding_size                    # embedding dimension
        self.emb_droprate = config.embedding_droprate           # embedding dropout rate
        self.seq_len = config.sequence_length                   # sequence length
        self.filter_count = config.filter_count                 # output feature size
        self.kernel_size = config.kernel_size                   # list of kernel size, means kGram in text, ex. [1, 2, 3, 4, 5 ...]
        self.conv_droprate = config.conv_droprate               # conventional layer dropout rate
        self.num_class = config.labels                          # classes

        pass

    def build(self):
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb_dropout = nn.Dropout(self.emb_droprate)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.filter_count, (k, self.emb_dim)) for k in self.kernel_size])
        self.pools = nn.ModuleList([nn.MaxPool2d((self.seq_len - k + 1, 1)) for k in self.kernel_size])
        self.conv_dropout = nn.Dropout(self.conv_droprate)
        self.fc = nn.Linear(len(self.kernel_size) * self.filter_count, self.num_class)
        pass

    def forward(self, input_):
        batch_size = input_.shape[0]

        x = self.embedding(input_)
        x = self.emb_dropout(x)

        # [batch, 1, length, dimension]
        x = torch.unsqueeze(x, dim=1)
        convs = []
        for conv in self.convs:
            convs.append(conv(x))

        pools = []
        for i, pool in enumerate(self.pools):
            pools.append(pool(convs[i]))

        x = torch.cat(pools, dim=1)
        # [batch, filter * layers]
        x = torch.squeeze(torch.squeeze(x, dim=2), dim=2)
        # [batch, 1, filter * layers]
        x = torch.unsqueeze(x, dim=1)
        x = self.conv_dropout(x)
        x = F.relu(x)

        # [batch, 1, num_class]
        x = self.fc(x)
        # [batch, num_class]
        x = torch.squeeze(x, dim=1)

        return x
