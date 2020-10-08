# 3rd-party module
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Model(torch.nn.Module):
    def __init__(self, config, num_embeddings,
                 padding_idx, embedding_weight=None):
        super(Model, self).__init__()

        # config
        self.config = config

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=config.embedding_dim,
                                            padding_idx=padding_idx)
        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # TS-BiLSTM
        self.TS = TSBiLSTM(config)

        # Linear layer
        self.linear = Linear(config)

    def forward(self, batch_x1, batch_x2):
        # Embedding
        batch_x1 = self.embedding_layer(batch_x1)
        batch_x2 = self.embedding_layer(batch_x2)

        # TS-BiLSTM
        batch_x, batch_weight = self.TS(batch_x1, batch_x2)

        # Linear layer
        batch_x = self.linear(batch_x)

        # Softmax
        batch_x = F.softmax(batch_x, dim=1)

        return batch_x, batch_weight

class TSBiLSTM(torch.nn.Module):
    def __init__(self, config):
        super(TSBiLSTM, self).__init__()

        # config
        self.config = config

        # target BiLSTM
        self.target_BiLSTM = nn.LSTM(input_size=config.embedding_dim,
                                     hidden_size=config.hidden_dim,
                                     num_layers=config.num_rnn_layers,
                                     batch_first=True,
                                     dropout=config.dropout,
                                     bidirectional=True)
        # claim BiLSTM
        self.claim_BiLSTM = nn.LSTM(input_size=config.embedding_dim,
                                    hidden_size=config.hidden_dim,
                                    num_layers=config.num_rnn_layers,
                                    batch_first=True,
                                    dropout=config.dropout,
                                    bidirectional=True)

    def forward(self, batch_x1, batch_x2):
        # target: get the final ht of the last layer
        target_ht, _ = self.target_BiLSTM(batch_x1) # (B, S, 2xH)
        target_ht = target_ht[:, -1] # (B, 2xH)

        # claim: get the all ht of the last layer
        claim_ht, _ = self.claim_BiLSTM(batch_x2) # (B, S, 2xH)

        # get the attention weight
        if self.config.attention == 'dot':
            weight = torch.bmm(claim_ht, target_ht.unsqueeze(2)).squeeze(2) # (B, S)
            soft_weight = F.softmax(weight, dim=1) # (B, S)

        # get final representation
        final_r = torch.bmm(soft_weight.unsqueeze(1), claim_ht).squeeze(1) # (B, 2xH)

        return final_r, soft_weight

class Linear(torch.nn.Module):
    def __init__(self, config):
        super(Linear, self).__init__()

        # config
        self.config = config

        # linear layer
        linear = [nn.Linear(in_features=config.hidden_dim*2,
                            out_features=config.stance_output_dim)]

        for _ in range(int(config.num_linear_layers)-1):
            linear.append(nn.Dropout(config.dropout))
            linear.append(nn.Linear(in_features=config.stance_output_dim,
                                    out_features=config.stance_output_dim))

        self.linear_layer = nn.Sequential(*linear)

    def forward(self, batch_x):
        batch_x = self.linear_layer(batch_x)

        return batch_x