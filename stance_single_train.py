# built-in module
import os
import pickle
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import tensorboard
import sklearn

# self-made module
import configs
import datas
import tokenizer
import embeddings
import models_single as models
import loss_single as loss

# hyperparameter setting
experiment_no = 1
config = configs.Config(stance_dataset='semeval2016',
                        embedding_file='glove/glove.twitter.27B.100d.txt',
                        random_seed=7,
                        epoch=30,
                        batch_size=32,
                        learning_rate=1e-4,
                        kfold=5,
                        dropout=0.2,
                        embedding_dim=100,
                        hidden_dim=100,
                        stance_output_dim=3,  # 3 for SemEval, 4 for FNC-1
                        num_rnn_layers=1,
                        num_linear_layers=1,
                        attention='dot',
                        clip_grad_value=0,
                        lexicon_loss_weight=0)

# deinfe save path
save_path = f'model_single/{experiment_no}'
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    raise ValueError(f'experiment {experiment_no} have already exist')

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
    data_df = datas.load_dataset('semeval2016_train')
elif config.stance_dataset == 'fnc-1':
    data_df = datas.load_dataset('fnc_train')

# initialize tokenizer and embedding
tokenizer = tokenizer.Tokenizer()
embedding = embeddings.Embedding(config.embedding_dim,
                                 config.random_seed)

# get all tokens and embeddings
all_sentence = []
all_sentence.extend(data_df['target'].drop_duplicates().tolist())
all_sentence.extend(data_df['claim'].drop_duplicates().tolist())

tokenizer.get_all_tokens(all_sentence)
embedding.load_embedding(f'data/embedding/{config.embedding_file}',
                         tokenizer.all_tokens)

# build vocabulary dictionary
tokenizer.build_dict(all_sentence, embedding.word_dict)

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

# save config, tokenizer and embedding
config_path = f'{save_path}/config.pickle'
tokenizer_path = f'{save_path}/tokenizer.pickle'
embedding_path = f'{save_path}/embedding.pickle'

config.save_to_file(config_path)
tokenizer.save_to_file(tokenizer_path)
embedding.save_to_file(embedding_path)

# define evaluate function
def evaluate(model, batch_iterator, phase='train'):
    total_loss = 0.0
    all_label_y, all_pred_y = [], []

    model.eval()
    with torch.no_grad():
        for _, x1, x2, lexicon, y in batch_iterator:
            # fed data into model
            x1 = x1.to(device)
            x2 = x2.to(device)
            pred_y, attn_weight = model(x1, x2)

            # evaluate loss
            batch_loss = loss.loss_function(lexicon_vector=lexicon,
                                            tokenizer=tokenizer, predict=pred_y,
                                            target=y, attn_weight=attn_weight,
                                            beta=config.lexicon_loss_weight,
                                            device=device)
            total_loss += batch_loss

            all_label_y.extend(y.tolist())
            all_pred_y.extend(torch.argmax(pred_y, axis=1).cpu().tolist())

    # evaluate loss
    total_loss = total_loss / len(batch_iterator)

    # evaluate f1
    if config.stance_dataset == 'semeval2016':  
        # semeval2016 benchmark just consider f1 score for "favor (0)" and "against (1)" label
        f1 = f1_score(all_label_y, all_pred_y, average='macro', labels=[0, 1])
    else:
        f1 = f1_score(all_label_y, all_pred_y, average='macro')

    return total_loss, f1

# initialize loss and f1
all_train_loss, all_train_f1 = [], []
all_valid_loss, all_valid_f1 = [], []
best_train_loss, best_train_f1 = None, None
best_valid_loss, best_valid_f1 = None, None
best_fold, best_epoch = None, None

# define KFold
kf = KFold(n_splits=config.kfold, random_state=config.random_seed,
           shuffle=True)
data_kf = kf.split(data_df)

for fold, (train_index, valid_index) in enumerate(data_kf, start=1):
    print(f'{fold}-fold\n')

    # DataFrame
    train_df = data_df.iloc[train_index]
    valid_df = data_df.iloc[valid_index]

    # SingleTask Dataset
    train_dataset = datas.SingleTaskDataset(
        task_id=0,
        target_encode=train_df['target_encode'],
        claim_encode=train_df['claim_encode'],
        claim_lexicon=train_df['claim_lexicon'],
        label_encode=train_df['label_encode'])
    valid_dataset = datas.SingleTaskDataset(
        task_id=0,
        target_encode=valid_df['target_encode'],
        claim_encode=valid_df['claim_encode'],
        claim_lexicon=valid_df['claim_lexicon'],
        label_encode=valid_df['label_encode'])

    # SingleTask Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=datas.SingleTaskDataset.collate_fn)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=datas.SingleTaskDataset.collate_fn)

    # construct model and optimizer
    model = models.Model(config=config,
                         num_embeddings=embedding.get_num_embeddings(),
                         padding_idx=tokenizer.pad_token_id,
                         embedding_weight=embedding.vector)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)

    # initialize loss and f1
    fold_train_loss, fold_train_f1 = [], []
    fold_valid_loss, fold_valid_f1 = [], []

    # train model
    model.zero_grad()

    for epoch in range(int(config.epoch)):
        model.train()
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc=f'epoch {epoch}', position=0)

        for _, train_x1, train_x2, \
            train_lexicon, train_y in train_iterator:
            # fed data in into model
            train_x1 = train_x1.to(device)
            train_x2 = train_x2.to(device)
            pred_y, attn_weight = model(train_x1, train_x2)

            # clean up gradient
            optimizer.zero_grad()

            # calculate loss
            batch_loss = loss.loss_function(lexicon_vector=train_lexicon,
                                            tokenizer=tokenizer, predict=pred_y,
                                            target=train_y, attn_weight=attn_weight,
                                            beta=config.lexicon_loss_weight,
                                            device=device)

            # backward pass
            batch_loss.backward()

            # prevent gradient boosting or vanishing
            if config.clip_grad_value != 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                                                config.clip_grad_value)

            # gradient decent
            optimizer.step()

        # evaluate model
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc='evaluate train set', position=0)
        valid_iterator = tqdm(valid_dataloader,
                              total=len(valid_dataloader),
                              desc='evaluate valid set', position=0)

        train_loss, train_f1 = evaluate(model, train_iterator, 'train')
        valid_loss, valid_f1 = evaluate(model, valid_iterator, 'test')

        print(f'train loss: {train_loss}, train f1: {train_f1}, '
              f'stance loss: {valid_loss}, stance f1: {valid_f1}\n')

        # save model
        if best_valid_loss is None or valid_loss < best_valid_loss:
            # check model save path
            if not os.path.exists(f'{save_path}/{fold}-fold'):
                os.makedirs(f'{save_path}/{fold}-fold')
            torch.save(model.state_dict(),
                       f'{save_path}/{fold}-fold/model_{epoch}.ckpt')

            best_train_loss, best_train_f1 = train_loss, train_f1
            best_valid_loss, best_valid_f1 = valid_loss, valid_f1
            best_fold, best_epoch = fold, epoch

        fold_train_loss.append(train_loss.item())
        fold_train_f1.append(train_f1.item())
        fold_valid_loss.append(valid_loss.item())
        fold_valid_f1.append(valid_f1.item())

    all_train_loss.append(fold_train_loss)
    all_train_f1.append(fold_train_f1)
    all_valid_loss.append(fold_valid_loss)
    all_valid_f1.append(fold_valid_f1)

# print final result
print(f'\n{best_fold}-fold epoch {best_epoch} - best train loss: {best_train_loss}, '
      f'best valid loss: {best_valid_loss}, best valid f1: {best_valid_f1}')

# init tensorboard
writer = SummaryWriter(f'tensorboard_single/experiment-{experiment_no}')

# write loss and f1 to tensorboard
for epoch in range(int(config.epoch)):
    writer.add_scalars(f'Loss/train',
                       {f'{fold}-fold': round(all_train_loss[fold][epoch], 5)
                       for fold in range(int(config.kfold))}, epoch)
    writer.add_scalars(f'F1/train',
                       {f'{fold}-fold': round(all_train_f1[fold][epoch], 3)
                       for fold in range(int(config.kfold))}, epoch)
    writer.add_scalars(f'Loss/valid',
                       {f'{fold}-fold': round(all_valid_loss[fold][epoch], 5)
                       for fold in range(int(config.kfold))}, epoch)
    writer.add_scalars(f'F1/valid',
                       {f'{fold}-fold': round(all_valid_f1[fold][epoch], 3)
                       for fold in range(int(config.kfold))}, epoch)

# add hyperparameters and final result to tensorboard
writer.add_hparams({
    'fold': str(best_fold),
    'epoch': str(best_epoch),
    'train_loss': best_train_loss.item(),
    'train_f1': best_train_f1.item(),
    'valid_loss': best_valid_loss.item(),
    'valid_f1': best_valid_f1.item(),
    'stance_dataset': config.stance_dataset,
    'embedding_file': config.embedding_file,
    'batch_size': str(config.batch_size),
    'lr': str(config.learning_rate),
    'dropout': str(config.dropout),
    'embedding_dim': str(config.embedding_dim),
    'hidden_dim': str(config.hidden_dim),
    'num_rnn_layers': str(config.num_rnn_layers),
    'num_linear_layers': str(config.num_linear_layers),
    'clip_grad_value': str(config.clip_grad_value),
    'lexicon_loss_weight': str(config.lexicon_loss_weight)
}, metric_dict={})
writer.close()

# release GPU memory
torch.cuda.empty_cache()