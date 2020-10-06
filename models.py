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
        self.stance_TS = TSBiLSTM(config)
        self.nli_TS = TSBiLSTM(config)
        self.shared_TS = TSBiLSTM(config)

        # Linear layer
        self.stance_linear = nn.Linear(in_features=config.hidden_dim*4,
                                       out_features=config.stance_output_dim)
        self.nli_linear = nn.Linear(in_features=config.hidden_dim*4,
                                    out_features=config.nli_output_dim)

        # self.stance_linear = Linear(config=config, task_id=0)
        # self.nli_linear = Linear(config=config, task_id=1)

    def forward(self, task_id, batch_x1, batch_x2):
        # Embedding
        batch_x1 = self.embedding_layer(batch_x1)
        batch_x2 = self.embedding_layer(batch_x2)

        # TS-BiLSTM
        # task_id: 0 for stance detection, 1 for NLI
        if task_id == 0:
            task_r, task_weight = self.stance_TS(batch_x1, batch_x2)
        elif task_id == 1:
            task_r, task_weight = self.nli_TS(batch_x1, batch_x2)
        shared_r, shared_weight = self.shared_TS(batch_x1, batch_x2)

        # Linear layer
        # task_id: 0 for stance detection, 1 for NLI
        task_r = torch.cat([task_r, shared_r], dim=1)
        if task_id == 0:
            task_r = self.stance_linear(task_r)
        elif task_id == 1:
            task_r = self.nli_linear(task_r)

        # Dropout and Softmax
        # task_r = F.softmax(F.dropout(task_r, p=self.config.dropout))
        task_r = F.softmax(task_r, dim=1)

        return task_r, (task_weight, shared_weight)

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
    def __init__(self, config, task_id):
        super(Linear, self).__init__()

        # config
        self.config = config

        # output dimension depends on task_id
        if task_id == 0:  # stance detection
            self.output_dim = config.stance_output_dim
        elif task_id == 1:  # NLI
            self.output_dim = config.nli_output_dim

        # linear layer
        linear = [nn.Linear(in_features=config.hidden_dim,
                            out_features=self.output_dim)]

        for _ in range(int(config.num_linear_layers)-1):
            linear.append(nn.Dropout(config.dropout))
            linear.append(nn.Linear(in_features=self.output_dim,
                                    out_features=self.output_dim))

        self.linear_layer = nn.Sequential(*linear)

    def forward(self, batch_x):
        batch_x = self.linear_layer(batch_x)

        return batch_x