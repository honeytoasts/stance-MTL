# 3rd-party module
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# self-made module
import tokenizers

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
                                       out_features=config.output_dim)
        self.nli_linear = nn.Linear(in_features=config.hidden_dim*4,
                                    out_features=config.output_dim)

    def foward(self, task_num, batch_x1, batch_x2):
        # Embedding
        batch_x1 = self.embedding_layer(batch_x1)
        batch_x2 = self.embedding_layer(batch_x2)

        # TS-BiLSTM
        # task_num: 0 for stance detection, 1 for NLI
        if task_num == 0:
            task_r, task_weight = self.stance_TS(batch_x1. batch_x2)
        elif task_num == 1:
            task_r, task_weight = self.nli_TS(batch_x1, batch_x2)
        shared_r, shared_weight = self.shared_TS(batch_x1, batch_x2)

        # Linear layer
        # task_num: 0 for stance detection, 1 for NLI
        task_r = torch.cat([task_r, shared_r], dim=1)
        if task_num == 0:
            task_r = self.stance_linear(task_r)
        elif task_num == 1:
            task_r = self.nli_linear(task_r)

        # Dropout and Softmax
        # task_r = F.softmax(F.dropout(task_r, p=self.config.dropout))
        task_r = F.softmax(task_r)

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

    def foward(self, batch_x1, batch_x2):
        # target: get the final ht of the last layer
        _, (target_ht, _) = self.target_BiLSTM(batch_x1) # (B, Nx2, H)
        target_ht = target_ht.view(self.config.batch_size, self.config.num_rnn_layers, -1) # (B, N, 2xH)
        target_ht = target_ht[:, -1].squeeze(0) # (B, 2xH)

        # claim: get the all ht of the last layer
        claim_ht, _ = self.claim_BiLSTM(batch_x2) # (B, S, 2xH)

        # get the attention weight
        if self.config.attention == 'dot':
            weight = torch.bmm(claim_ht, target_ht.unsqueeze(2)).squeeze(2) # (B, S)
            soft_weight = F.softmax(weight, dim=1) # (B, S)

        # get final representation
        final_r = torch.bmm(soft_weight.unsqueeze(1), claim_ht).squeeze(1) # (B, 2xH)

        return final_r, soft_weight