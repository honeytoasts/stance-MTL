# 3rd-party module
import torch
from torch import nn
from torch.nn import functional as F
from util import custom_lstms

class BaseModel(torch.nn.Module):
    def __init__(self, config, num_embeddings,
                 padding_idx, embedding_weight=None):
        super(BaseModel, self).__init__()

        # config
        self.config = config

        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=config.embedding_dim,
                                            padding_idx=padding_idx)
        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # TS-BiLSTM
        self.stance_TS = TSBiLSTM(config, hidden_dim=config.task_hidden_dim)
        self.nli_TS = TSBiLSTM(config, hidden_dim=config.task_hidden_dim)
        self.shared_TS = TSBiLSTM(config, hidden_dim=config.shared_hidden_dim)

        # linear layer
        self.stance_linear = Linear(config, task_id=0)
        self.nli_linear = Linear(config, task_id=1)

    def forward(self, task_id, batch_x1, batch_x2):
        # task_id: 0 for stance detection, 1 for NLI

        # embedding
        batch_x1 = self.embedding_layer(batch_x1)
        batch_x2 = self.embedding_layer(batch_x2)

        # TS-BiLSTM
        if task_id == 0:
            task_r, task_weight = self.stance_TS(batch_x1, batch_x2)
        elif task_id == 1:
            task_r, task_weight = self.nli_TS(batch_x1, batch_x2)

        shared_r, shared_weight = self.shared_TS(batch_x1, batch_x2)

        # linear layer
        task_r = torch.cat([task_r, shared_r], dim=1)

        if task_id == 0:
            task_r = self.stance_linear(task_r)
        elif task_id == 1:
            task_r = self.nli_linear(task_r)

        # softmax
        task_r = F.softmax(task_r, dim=1)

        return task_r, task_weight, shared_weight

class TSBiLSTM(torch.nn.Module):
    def __init__(self, config, hidden_dim):
        super(TSBiLSTM, self).__init__()

        # config
        self.config = config

        # get parameters of LSTM
        target_parameter = {'input_size': config.embedding_dim,
                            'hidden_size': hidden_dim,
                            'num_layers': 1,
                            'batch_first': True,
                            'bidirectional': True}
        claim_parameter = {'input_size': config.embedding_dim,
                           'hidden_size': hidden_dim,
                           'num_layers': config.num_rnn_layers,
                           'batch_first': True,
                           'bidirectional': True}
        if int(config.num_rnn_layers) > 1:
            claim_parameter['dropout'] = config.dropout

        # target BiLSTM
        self.target_BiLSTM = nn.LSTM(**target_parameter)

        # claim BiLSTM
        self.claim_BiLSTM = nn.LSTM(**claim_parameter)

        # linear layer for attention
        if config.attention == 'linear':
            self.attn_linear = nn.Linear(hidden_dim * 4, 1)  # to scalar value

    def forward(self, batch_x1, batch_x2):
        # get claim sequence length
        claim_seq_len = batch_x2.shape[1]

        # target: get final ht of the last layer
        target_ht, _ = self.target_BiLSTM(batch_x1)  # (B, S, 2H)
        target_ht = target_ht[:, -1]  # (B, 2H)

        # claim: get all ht of the last layer
        claim_ht, _ = self.claim_BiLSTM(batch_x2)  # (B, S, 2H)

        # get the attention weight
        if self.config.attention == 'dot':
            weight = torch.matmul(claim_ht, target_ht.unsqueeze(2)).squeeze(2)  # (B, S)
            soft_weight = F.softmax(weight, dim=1)  # (B, S)

        elif self.config.attention == 'linear':
            # get target hidden vector
            target_ht = target_ht.repeat(claim_seq_len, 1, 1)  # (S, B, 2H)
            target_ht = target_ht.transpose(0, 1)  # (B, S, 2H)

            # concat target and claim
            new_claim_ht = torch.cat((target_ht, claim_ht), 2)  # (B, S, 4H)

            # apply linear layer to get the attention weight
            weight = self.attn_linear(new_claim_ht)  # (B, S, 1)
            soft_weight = F.softmax(weight.squeeze(2), dim=1)  # (B, S)

        elif self.config.attention == 'cosine':
            # get target hidden vector
            target_ht = target_ht.repeat(claim_seq_len, 1, 1)  # (S, B, 2H)
            target_ht = target_ht.transpose(0, 1)  # (B, S, 2H)

            # calculate cosine similarity to get attention weight
            weight = F.cosine_similarity(target_ht, claim_ht, dim=2)  # (B, S)
            soft_weight = F.softmax(weight, dim=1)  # (B, S)

        # get final representation
        final_r = torch.matmul(soft_weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, 2H)

        return final_r, soft_weight

class Linear(torch.nn.Module):
    def __init__(self, config, task_id):
        super(Linear, self).__init__()
        # get input dimension
        input_dim = config.task_hidden_dim * 2 + config.shared_hidden_dim * 2

        # get linear and output dimension
        if task_id == 0:  # stance detection
            linear_dim = config.stance_linear_dim
            output_dim = config.stance_output_dim
        elif task_id == 1:  # NLI
            linear_dim = config.nli_linear_dim
            output_dim = config.nli_output_dim

        # linear layer
        linear = [nn.Linear(in_features=input_dim,
                            out_features=linear_dim)]

        for _ in range(config.num_linear_layers-2):
            linear.append(nn.ReLU())
            linear.append(nn.Dropout(config.linear_dropout))
            linear.append(nn.Linear(in_features=linear_dim,
                                    out_features=linear_dim))

        linear.append(nn.ReLU())
        linear.append(nn.Dropout(config.linear_dropout))
        linear.append(nn.Linear(in_features=linear_dim,
                                out_features=output_dim))

        self.linear = nn.Sequential(*linear)

    def forward(self, task_r):
        task_r = self.linear_layer(task_r)

        return task_r