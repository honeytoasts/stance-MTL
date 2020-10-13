# built-in module
import os
import pickle
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import sklearn

# self-made module
import configs
import datas
import tokenizer
import embeddings
import models
import loss

# parameter and model path setting
experiment_no = 1
fold = 1
epoch = 39
model_path = f'model/{experiment_no}/'

# initialize device
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

# load config, tokenizer, embedding
config = configs.Config()
config.load_from_file(f'{model_path}/config.pickle')

tokenizer = tokenizer.Tokenizer()
tokenizer.load_from_file(f'{model_path}/tokenizer.pickle')

embedding = embeddings.Embedding()
embedding.load_from_file(f'{model_path}/embedding.pickle')

# load data
if config.stance_dataset == 'semeval2016':
    data_df = datas.load_dataset('semeval2016_test')
elif config.stance_dataset == 'fnc-1':
    data_df = datas.load_dataset('fnc_test')

# content encode to id
print('content encode --')
data_df['target_encode'] = \
    tokenizer.encode(data_df['target'].tolist())
data_df['claim_encode'] = \
    tokenizer.encode(data_df['claim'].tolist())

# label encode
print('label encode --')
if 'semeval' in config.stance_dataset:
    stance_label = {'favor': 0, 'against': 1, 'none': 2}
elif 'fnc' in config.stance_dataset:
    stance_label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

data_df['label_encode'] = data_df['label'].apply(
    lambda label: stance_label[label])

# load lexicon
lexicon = datas.load_lexicon(lexicon='emolex_emotion')

# content encode to lexicon vector
print('lexicon encode --')
data_df['claim_lexicon'] = \
    tokenizer.encode_to_lexicon(data_df['claim'].tolist(), lexicon)

# define evaluate function
def evaluate(model, batch_iterator):
    total_loss, total_lexicon_loss = 0.0, 0.0
    all_label_y, all_pred_y = [], []

    model.eval()
    with torch.no_grad():
        for task_id, x1, x2, lexicon, y in batch_iterator:
            # fed data into model
            x1 = x1.to(device)
            x2 = x2.to(device)
            pred_y, attn_weight = model(task_id, x1, x2)

            # evaluate loss
            batch_loss, batch_lexicon_loss = \
                loss.loss_function(lexicon_vector=lexicon,
                                   tokenizer=tokenizer, predict=pred_y,
                                   target=y, attn_weight=attn_weight,
                                   beta=config.lexicon_loss_weight,
                                   device=device)
            total_loss += batch_loss
            total_lexicon_loss += batch_lexicon_loss

            all_label_y.extend(y.tolist())
            all_pred_y.extend(torch.argmax(pred_y, axis=1).cpu().tolist())

    # evaluate loss
    total_loss = total_loss / len(batch_iterator)
    total_lexicon_loss = total_lexicon_loss / len(batch_iterator)

    # evaluate f1
    if config.stance_dataset == 'semeval2016':  
        # semeval2016 benchmark just consider f1 score for "favor (0)" and "against (1)" label
        f1 = f1_score(all_label_y, all_pred_y, average='macro', labels=[0, 1])
    else:
        f1 = f1_score(all_label_y, all_pred_y, average='macro')

    return (total_loss, total_lexicon_loss, f1,
            all_label_y, all_pred_y)

# define dataset and dataloader
dataset = datas.SingleTaskDataset(
    task_id=0,
    target_encode=data_df['target_encode'],
    claim_encode=data_df['claim_encode'],
    claim_lexicon=data_df['claim_lexicon'],
    label_encode=data_df['label_encode'])
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=datas.SingleTaskDataset.collate_fn)

# load model
model = models.Model(config=config,
                     num_embeddings=embedding.get_num_embeddings(),
                     padding_idx=tokenizer.pad_token_id,
                     embedding_weight=embedding.vector)
model.load_state_dict(
    torch.load(f'{model_path}/{fold}-fold/model_{epoch}.ckpt'))
model = model.to(device)

# evaluate
model.eval()

test_iterator = tqdm(dataloader, total=len(dataloader),
                     desc='evaluate test set', position=0)
loss, lexicon_loss, f1, label_y, pred_y = \
    evaluate(model, test_iterator)

# print result
print(f'\nexperiment {experiment_no}: {fold}-fold {epoch}-epoch\n'
      f'dataset: {config.stance_dataset}\n'
      f'loss: {round(loss.item(), 5)}, '
      f'lexicon loss: {round(lexicon_loss.item(), 5)}\n'
      f'f1 score: {round(f1.item(), 5)}')