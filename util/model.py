# 3rd-party module
import argparse
import torch
from torch import nn
from torch.nn import functional as F

class TaskSpecificSharedModel(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace,
                 num_embeddings: int,
                 padding_idx: int,
                 embedding_weight=None):
        super(TaskSpecificSharedModel, self).__init__()

        # config
        self.config = config

        # dropout layer
        self.rnn_dropout = nn.Dropout(config.rnn_dropout)
        self.gcn_dropout = nn.Dropout(config.gcn_dropout)

        # embedding layer
        self.embedding_layer = (
            nn.Embedding(num_embeddings=num_embeddings,
                         embedding_dim=config.embedding_dim,
                         padding_idx=padding_idx))
        if embedding_weight is not None:
            self.embedding_layer.weight = nn.Parameter(embedding_weight)

        # StanceAttn layer
        self.stance_Attn = StanceAttn(config=config,
                                      hidden_dim=config.stance_hidden_dim,
                                      num_layers=config.num_stance_rnn)
        self.nli_Attn = StanceAttn(config=config,
                                   hidden_dim=config.nli_hidden_dim,
                                   num_layers=config.num_nli_rnn)
        self.shared_Attn = StanceAttn(config=config,
                                      hidden_dim=config.shared_hidden_dim,
                                      num_layers=config.num_shared_rnn)

        # StanceGCN layer
        self.stance_GCN = StanceGCN(config=config,
                                    hidden_dim=2*config.stance_hidden_dim,
                                    num_layers=config.num_stance_gcn)
        self.nli_GCN = StanceGCN(config=config,
                                 hidden_dim=2*config.nli_hidden_dim,
                                 num_layers=config.num_nli_gcn)
        self.shared_GCN = StanceGCN(config=config,
                                    hidden_dim=2*config.shared_hidden_dim,
                                    num_layers=config.num_shared_gcn)

        # StanceLinear layer
        stance_input_dim = (config.stance_hidden_dim*2 +
                            config.shared_hidden_dim*2)
        nli_input_dim = (config.nli_hidden_dim*4 +
                         config.shared_hidden_dim*4)

        self.stance_linear = (
            StanceLinear(config=config,
                         input_dim=stance_input_dim,
                         hidden_dim=config.stance_linear_dim,
                         output_dim=config.stance_output_dim,
                         num_layers=config.num_stance_linear))
        self.nli_linear = (
            NliLinear(config=config,
                         input_dim=nli_input_dim,
                         hidden_dim=config.nli_linear_dim,
                         output_dim=config.nli_output_dim,
                         num_layers=config.num_nli_linear))

    def forward(self,
                task_id: int,
                task_target,
                shared_target,
                task_claim,
                shared_claim,
                task_mask,
                shared_mask,
                task_adj,
                shared_adj):

        # embedding
        task_target = self.embedding_layer(task_target)
        shared_target = self.embedding_layer(shared_target)

        task_claim = self.embedding_layer(task_claim)
        shared_claim = self.embedding_layer(shared_claim)

        # dropout
        # task_target = self.rnn_dropout(task_target)
        shared_target = self.rnn_dropout(shared_target)

        # task_claim = self.rnn_dropout(task_claim)
        shared_claim = self.rnn_dropout(shared_claim)

        # StanceAttn
        task_target, task_claim, task_weight = (
            self.stance_Attn(task_target, task_claim, task_mask)
            if task_id == 0 else
            self.nli_Attn(task_target, task_claim, task_mask))

        shared_target, shared_claim, shared_weight = (
            self.shared_Attn(shared_target, shared_claim, shared_mask))

        # dropout
        task_target = self.rnn_dropout(task_target)
        shared_target = self.rnn_dropout(shared_target)

        task_claim = self.rnn_dropout(task_claim)
        shared_claim = self.rnn_dropout(shared_claim)

        # get target and claim sequence length
        task_target_len = task_target.shape[1]
        shared_target_len = shared_target.shape[1]

        task_claim_len = task_claim.shape[1]
        shared_claim_len = shared_claim.shape[1]

        # add edge to adjacency matrix between the target and claim
        # with attention weight larger than threshold
        task_adj_edge = (task_weight >= self.config.attention_threshold)  # (B, S)
        task_adj_edge = task_adj_edge.repeat(1, task_target_len).reshape(
            -1, task_target_len, task_claim_len)  # (B, S, S)

        shared_adj_edge = (shared_weight >= self.config.attention_threshold)  # (B, S)
        shared_adj_edge = shared_adj_edge.repeat(1, shared_target_len).reshape(
            -1, shared_target_len, shared_claim_len)  # (B, S, S)

        task_adj[:, :task_target_len, task_target_len:] = task_adj_edge
        task_adj[:, task_target_len:, :task_target_len] = (
            task_adj_edge.transpose(1, 2))

        shared_adj[:, :shared_target_len, shared_target_len:] = shared_adj_edge
        shared_adj[:, shared_target_len:, :shared_target_len] = (
            shared_adj_edge.transpose(1, 2))

        # StanceGCN
        task_target, task_claim, task_node_rep, task_corr_score = (
            self.stance_GCN(task_target, task_claim, task_adj)
            if task_id == 0 else
            self.nli_GCN(task_target, task_claim, task_adj))

        shared_target, shared_claim, shared_node_rep, shared_corr_score = (
            self.shared_GCN(shared_target, shared_claim, shared_adj))

        # dropout
        task_target = self.gcn_dropout(task_target)
        shared_target = self.gcn_dropout(shared_target)

        task_claim = self.gcn_dropout(task_claim)
        shared_claim = self.gcn_dropout(shared_claim)

        # linear layer
        final_rep = (
            self.stance_linear(task_claim, task_weight,
                               shared_claim, shared_weight)
            if task_id == 0 else
            self.nli_linear(task_target, task_claim, task_weight,
                            shared_target, shared_claim, shared_weight))

        return (final_rep, 
                (task_weight, shared_weight,
                 task_node_rep, shared_node_rep,
                 task_corr_score, shared_corr_score))

class StanceAttn(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace,
                 hidden_dim: int,
                 num_layers: int):
        super(StanceAttn, self).__init__()

        # config
        self.config = config

        # parameters of GRU
        parameters = {'input_size': config.embedding_dim,
                      'hidden_size': hidden_dim,
                      'num_layers': num_layers,
                      'batch_first': True,
                      'dropout': config.rnn_dropout
                                 if num_layers > 1 else 0,
                      'bidirectional': True}

        # target BiGRU
        self.target_BiGRU = nn.GRU(**parameters)

        # claim BiGRU
        self.claim_BiGRU = nn.GRU(**parameters)

        # linear layer for attention
        if config.attention == 'linear':

            # linear transformation for target
            self.t_linear = nn.Linear(in_features=2*hidden_dim,
                                      out_features=2*hidden_dim,
                                      bias=False)

            # linear transformation for claim
            self.c_linear = nn.Linear(in_features=2*hidden_dim,
                                      out_features=2*hidden_dim,
                                      bias=False)

            # linear transformation for attention score
            self.attn_linear = nn.Linear(in_features=2*hidden_dim,
                                         out_features=1,
                                         bias=False)

    def forward(self,
                batch_target,
                batch_claim,
                batch_mask):

        # get claim sequence length
        claim_len = batch_claim.shape[1]

        # get target hidden state
        target_ht, _ = self.target_BiGRU(batch_target)  # (B, S, H)
        target_last_ht = target_ht[:, -1]  # (B, H)

        # get all claim hidden state
        claim_ht, _ = self.claim_BiGRU(batch_claim)  # (B, S, H)

        # get attention weight
        if self.config.attention == 'dot':
            # matrix product
            weight = torch.matmul(claim_ht, 
                                  target_last_ht.unsqueeze(2)).squeeze(2)  # (B, S)

        elif self.config.attention == 'linear':
            # linear transformation
            e = torch.tanh(self.t_linear(target_last_ht).unsqueeze(1) +   # (B, 1, H)
                           self.c_linear(claim_ht))  # (B, S, H)

            # get attention score
            weight = self.attn_linear(e).squeeze(2)  # (B, S)

        elif self.config.attention == 'cosine':
            # get target hidden vector
            target_last_ht = target_last_ht.repeat(claim_len, 1, 1)  # (S, B, H)
            target_last_ht = target_last_ht.transpose(0, 1)  # (B, S, H)

            # apply cosine similarity to get attention weight
            weight = F.cosine_similarity(target_last_ht, claim_ht, dim=2)  # (B, S)

        # attention mask
        weight.masked_fill(batch_mask == 0, -1e9)

        # get attention softmax weight
        soft_weight = F.softmax(weight, dim=1)  # (B, S)

        return target_ht, claim_ht, soft_weight  # (B, S, H), (B, S, H), (B, S)

class StanceGCN(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace,
                 hidden_dim: int,
                 num_layers: int):
        super(StanceGCN, self).__init__()

        # config
        self.config = config

        # num of gcn layers
        self.num_layers = num_layers

        # GCN layer
        GCN = nn.ModuleList()

        for _ in range(num_layers):
            GCN.append(StanceGCNLayer(config=config,
                                      in_features=hidden_dim,
                                      out_features=hidden_dim))

        self.GCN = GCN

        # dropout layer
        self.gcn_dropout = nn.Dropout(config.gcn_dropout)

    def forward(self,
                batch_target,
                batch_claim,
                batch_adj):

        # get target sequence kength
        target_len = batch_target.shape[1]

        # get node representation by concat target and claim representation
        node_rep = torch.cat([batch_target, batch_claim], dim=1)  # (B, S, H)

        all_node_rep, all_corr_score = (), ()

        # iterate all the GCN layers
        for i, GCN in enumerate(self.GCN):
            node_rep, corr_score = GCN(node_rep, batch_adj)

            # record representaion and correlation score
            all_node_rep = all_node_rep + (node_rep,)
            all_corr_score = all_corr_score + (corr_score,)

            # apply dropout if not the last layer
            if i != self.num_layers-1:
                node_rep = self.gcn_dropout(node_rep)

        # get target and claim representation
        target_rep = node_rep[:, :target_len, :]
        claim_rep = node_rep[:, target_len:, :]

        return target_rep, claim_rep, all_node_rep, all_corr_score

class StanceGCNLayer(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace,
                 in_features: int,
                 out_features: int):
        super(StanceGCNLayer, self).__init__()

        # config
        self.config = config

        # dimension
        self.in_features = in_features
        self.out_features = out_features

        assert in_features == out_features

        # linear layer for correlation score
        if config.attention == 'linear':
            self.corr_linear1 = (
                nn.Linear(2*in_features, 2*in_features, bias=False))
            self.corr_linear2 = (
                nn.Linear(2*in_features, 1, bias=False))

        # linear layer for linear transformation
        self.trans_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self,
                node_rep,
                batch_adj):

        # get sequence length
        total_len = node_rep.shape[1]

        # get node i and node j
        """
        | h1 h1 h1 |
        | h2 h2 h2 |
        | h3 h3 h3 |
        """
        node_hi = node_rep.repeat(1, 1, total_len).reshape(
            -1, total_len, total_len, self.in_features)  # (B, S, S, H)
        """
        | h1 h2 h3 |
        | h1 h2 h3 |
        | h1 h2 h3 |
        """
        node_hj = node_hi.transpose(1, 2)  # (B, S, S, H)

        # get correlation score between node i and j
        if self.config.attention == 'dot':
            corr_score = torch.matmul(
                node_hj, node_rep.unsqueeze(3)).squeeze(3)  # (B, S, S)
        
        elif self.config.attention == 'linear':
            # concatenation of node i and node j
            node_hij = torch.stack([node_hi, node_hj], dim=3).reshape(
                -1, total_len, total_len, 2*self.in_features)  # (B, S, S, 2H)

            # apply linear layer to get the correlation score
            corr_score = self.corr_linear2(
                torch.relu(self.corr_linear1(node_hij))).squeeze(3)  # (B, S, S)

        elif self.config.attention == 'cosine':
            corr_score = F.cosine_similarity(node_hi, node_hj, dim=3)  # (B, S, S)

        # mask for nodes are not connected
        corr_score.masked_fill(batch_adj == 0, -1e9)

        # get softmax score
        corr_score = F.softmax(corr_score, dim=2)  # (B, S, S)

        # linear transformation and multiply correlation score
        node_trans_rep = self.trans_linear(node_rep)  # (B, S, H)
        node_trans_rep = torch.matmul(corr_score, node_trans_rep)  # (B, S, H)

        # update node representation
        node_rep = torch.relu(node_rep + node_trans_rep)  # (B, S, H)

        return node_rep, corr_score

class StanceLinear(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int):
        super(StanceLinear, self).__init__()

        # config
        self.config = config

        # linear dropout
        self.linear_dropout = nn.Dropout(p=config.linear_dropout)

        # linear layer
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=output_dim)

    def forward(self,
                task_claim,
                task_weight,
                shared_claim,
                shared_weight):

        # get linear combination vector of claim
        task_rep = torch.matmul(task_weight.unsqueeze(1),
                                task_claim).squeeze(1)  # (B, H)
        shared_rep = torch.matmul(shared_weight.unsqueeze(1),
                                  shared_claim).squeeze(1)  # (B, H)
        final_rep = torch.cat([task_rep, shared_rep], dim=1)

        # dropout
        final_rep = self.linear_dropout(final_rep)

        # linear layer
        final_rep = self.linear(final_rep)

        return final_rep

class NliLinear(torch.nn.Module):
    def __init__(self,
                 config: argparse.Namespace,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int):
        super(NliLinear, self).__init__()

        # config
        self.config = config

        # linear dropout
        self.linear_dropout = nn.Dropout(p=config.linear_dropout)

        # linear layer
        self.linear = nn.Linear(in_features=input_dim,
                                out_features=output_dim)

    def forward(self,
                task_target,
                task_claim,
                task_weight,
                shared_target,
                shared_claim,
                shared_weight):

        # get average vector of target
        task_target = torch.mean(task_target, dim=1)
        shared_target = torch.mean(shared_target, dim=1)

        # get linear combination vector of claim
        task_claim = torch.matmul(task_weight.unsqueeze(1),
                                  task_claim).squeeze(1)  # (B, H)
        shared_claim = torch.matmul(shared_weight.unsqueeze(1),
                                    shared_claim).squeeze(1)  # (B, H)
        
        # get final representation of target and claim
        task_rep = torch.cat([task_target, task_claim], dim=1)
        shared_rep = torch.cat([shared_target, shared_claim], dim=1)
        final_rep = torch.cat([task_rep, shared_rep], dim=1)

        # dropout
        final_rep = self.linear_dropout(final_rep)

        # linear layer
        final_rep = self.linear(final_rep)

        return final_rep