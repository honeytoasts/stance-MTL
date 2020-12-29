# 3rd-party module
import torch
from torch import nn
from torch.nn import functional as F

class BaseModel(torch.nn.Module):
    def __init__(self, config, num_embeddings,
                 padding_idx, embedding_weight=None):
        super(BaseModel, self).__init__()

        # config
        self.config = config

        # Embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=config.embedding_dim,
                                            padding_idx=padding_idx)
        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # TS-BiLSTM
        self.stance_TS = TSBiLSTM(config, shared=False)
        self.nli_TS = TSBiLSTM(config, shared=False)
        self.shared_TS = TSBiLSTM(config, shared=True)

        # Linear layer
        self.stance_linear = Linear(config=config, task_id=0)
        self.nli_linear = Linear(config=config, task_id=1)

    def forward(self, task_id, batch_x1, batch_x2):
        # task_id: 0 for stance detection, 1 for NLI

        # Embedding
        batch_x1 = self.embedding_layer(batch_x1)
        batch_x2 = self.embedding_layer(batch_x2)

        # TS-BiLSTM
        if task_id == 0:
            task_r, task_weight = self.stance_TS(batch_x1, batch_x2)
        elif task_id == 1:
            task_r, task_weight = self.nli_TS(batch_x1, batch_x2)
        shared_r, shared_weight = self.shared_TS(batch_x1, batch_x2)

        # Linear layer
        task_r = torch.cat([task_r, shared_r], dim=1)

        if task_id == 0:
            task_r = self.stance_linear(task_r)
        elif task_id == 1:
            task_r = self.nli_linear(task_r)

        # Softmax
        task_r = F.softmax(task_r, dim=1)

        return task_r, (task_weight, shared_weight)

class TSBiLSTM(torch.nn.Module):
    def __init__(self, config, shared=False):
        super(TSBiLSTM, self).__init__()

        # config
        self.config = config

        # get hidden dimension
        if shared == False:
            hidden_dim = config.task_hidden_dim
        elif shared == True:
            hidden_dim = config.shared_hidden_dim

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
            # concat target and claim
            target_ht = target_ht.repeat_interleave(
                self.config.max_seq_len, 0)  # (BxS, 2H)
            target_ht = target_ht.reshape(
                -1, self.config.max_seq_len, target_ht.shape[1])  # (B, S, 2H)
            new_claim_ht = torch.cat((claim_ht, target_ht), 2)  # (B, S, 4H)

            # apply linear layer to get the attention weight
            weight = self.attn_linear(new_claim_ht)  # (B, S, 1)
            soft_weight = F.softmax(weight.squeeze(2), dim=1)  # (B, S)

        elif self.config.attention == 'cos':
            target_ht = target_ht.repeat_interleave(
                self.config.max_seq_len, 0)  # (BxS, 2H)
            target_ht = target_ht.reshape(
                -1, self.config.max_seq_len, target_ht.shape[1])  # (B, S, 2H)

            weight = F.cosine_similarity(claim_ht, target_ht, dim=2)  # (B, S)
            soft_weight = F.softmax(weight, dim=1)  # (B, S)

        # get final representation
        final_r = torch.matmul(soft_weight.unsqueeze(1), claim_ht).squeeze(1)  # (B, 2H)

        return final_r, soft_weight

class Linear(torch.nn.Module):
    def __init__(self, config, task_id):
        super(Linear, self).__init__()
        # get input dimension
        input_dim = config.task_hidden_dim * 2 + config.shared_hidden_dim * 2

        # get output dimension
        if task_id == 0:  # stance detection
            output_dim = config.stance_output_dim
        elif task_id == 1:  # NLI
            output_dim = config.nli_output_dim

        # linear layer
        linear = [nn.Linear(in_features=input_dim,
                            out_features=output_dim)]

        for _ in range(int(config.num_linear_layers)-1):
            linear.append(nn.Dropout(config.dropout))
            linear.append(nn.Linear(in_features=output_dim,
                                    out_features=output_dim))

        self.linear_layer = nn.Sequential(*linear)

    def forward(self, batch_x):
        batch_x = self.linear_layer(batch_x)

        return batch_x