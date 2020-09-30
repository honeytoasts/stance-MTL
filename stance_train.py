# built-in module
import os
import pickle
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# self-made module
import configs
import datas
import tokenizer
import embeddings
import models

# hyperparameter setting
experiment_no = 1
config = configs.Config(stance_dataset='semeval2016',
                        embedding_file='glove/glove.twitter.27B.100d.txt',
                        random_seed=7,
                        epoch=50,
                        batch_size=32,
                        learning_rate=1e-3,
                        kfold=5,
                        dropout=0.5,
                        embedding_dim=100,
                        hidden_dim=100,
                        stance_output_dim=3,
                        nli_output_dim=3,
                        num_rnn_layers=1,
                        num_linear_layers=1)

# initialize random seed and device
device = torch.device('cpu')
os.environ['PYTHONHASHSEED'] = str(config.random_seed)
random.seed(config.random_seed)
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# load data
if config.stance_dataset == 'semeval2016':
    stance_data_df = datas.load_dataset('semeval2016_train')
elif config.stance_dataset == 'fnc-1':
    stance_data_df = datas.load_dataset('fnc_train')

nli_data_df = datas.load_dataset('mnli_train')

# initialize tokenizer and embedding
tokenizer = tokenizer.Tokenizer()
embedding = embeddings.Embedding(config.embedding_dim,
                                 config.random_seed)

# get all tokens and embeddings
all_sentence = []
all_sentence.extend(stance_data_df['target'].drop_duplicates().tolist())
all_sentence.extend(stance_data_df['claim'].drop_duplicates().tolist())
all_sentence.extend(nli_data_df['target'].drop_duplicates().tolist())
all_sentence.extend(nli_data_df['claim'].drop_duplicates().tolist())

all_tokens = tokenizer.get_all_tokens(all_sentence)
embedding.load_embedding(f'data/embedding/{config.embedding_file}',
                         all_tokens)

# build vocabulary dictionary
tokenizer.build_dict(all_sentence, embedding.word_dict)

# content encode
stance_data_df['target_encode'] = \
    tokenizer.encode(stance_data_df['target'].tolist())
stance_data_df['claim_encode'] = \
    tokenizer.encode(stance_data_df['claim'].tolist())
nli_data_df['target_encode'] = \
    tokenizer.encode(nli_data_df['target'].tolist())
nli_data_df['claim_encode'] = \
    tokenizer.encode(nli_data_df['claim'].tolist())

# label encode
if 'semeval' in config.stance_dataset:
    stance_label = {'favor': 0, 'against': 1, 'none': 2}
elif 'fnc' in config.stance_dataset:
    stance_label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
nli_label = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

stance_data_df['label_encode'] = stance_data_df['label'].apply(
    lambda label: stance_label[label])
nli_data_df['label_encode'] = nli_data_df['label'].apply(
    lambda label: nli_label[label])

# define save path
save_path = f'data/model/{experiment_no}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

config_path = f'{save_path}/config.pickle'
tokenizer_path = f'{save_path}/tokenizer.pickle'
embedding_path = f'{save_path}/embedding.pickle'
model_path = f'{save_path}/model.ckpt'

config.save_to_file(config_path)
tokenizer.save_to_file(tokenizer_path)
embedding.save_to_file(embedding_path)

# initialize loss and f1
best_train_loss, best_valid_loss = None, None
best_train_f1, best_valid_f1 = None, None

# define KFold
kf = KFold(n_splits=config.kfold, random_state=config.random_seed,
           shuffle=True)
stance_kf = kf.split(stance_data_df)
nli_kf = kf.split(nli_data_df)

for (stance_train_index, stance_valid_index), \
    (nli_train_index, nli_valid_index) in zip(stance_kf, nli_kf):
    # DataFrame
    stance_train_df = stance_data_df[stance_train_index]
    stance_valid_df = stance_data_df[stance_valid_index]
    nli_train_df = nli_data_df[nli_train_index]
    nli_valid_df = nli_data_df[nli_valid_index]

    # SingleTask Dataset
    stance_train_dataset = datas.SingleTaskDataset(
        task_id=0,
        target_encode=stance_train_df['target_encode'],
        claim_encode=stance_train_df['claim_encode'],
        label_encode=stance_train_df['label_encode'])
    stance_valid_dataset = datas.SingleTaskDataset(
        task_id=0,
        target_encode=stance_valid_df['target_encode'],
        claim_encode=stance_valid_df['claim_encode'],
        label_encode=stance_valid_df['label_encode'])
    nli_train_dataset = datas.SingleTaskDataset(
        task_id=1,
        target_encode=nli_train_df['target_encode'],
        claim_encode=nli_train_df['claim_encode'],
        label_encode=nli_train_df['label_encode'])
    nli_valid_dataset = datas.SingleTaskDataset(
        task_id=1,
        target_encode=nli_valid_df['target_encode'],
        claim_encode=nli_valid_df['claim_encode'],
        label_encode=nli_valid_df['label_encode'])

    # MultiTask Dataset
    train_dataset = datas.MultiTaskDataset(
        [stance_train_dataset, nli_train_dataset])

    # SingleTask Dataloader
    stance_valid_dataloader = DataLoader(
        dataset=stance_valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=datas.SingleTaskDataset.collate_fn)
    nli_valid_dataloader = DataLoader(
        dataset=nli_valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=datas.SingleTaskDataset.collate_fn)

    # MultiTask DataLoader
    multitask_batch_sampler = datas.MultiTaskBatchSampler(
        [stance_train_dataset, nli_train_dataset],
        batch_size=config.batch_size,
        random_seed=config.random_seed)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=multitask_batch_sampler,
        collate_fn=datas.SingleTaskDataset.collate_fn)

    # construct model, loss function, optimizer