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
from tqdm import tqdm
import tensorboard
import sklearn

# self-made module
import configs
import datas
import tokenizer
import embeddings
import models
import loss

# hyperparameter setting
experiment_no = 1
config = configs.Config(stance_dataset='semeval2016',
                        embedding_file='glove/glove.twitter.27B.100d.txt',
                        random_seed=7,
                        epoch=30,
                        batch_size=64,
                        learning_rate=1e-3,
                        kfold=5,
                        dropout=0.2,
                        embedding_dim=100,
                        hidden_dim=100,
                        stance_output_dim=3,  # 3 for SemEval, 4 for FNC-1
                        nli_output_dim=3,
                        num_rnn_layers=1,
                        num_linear_layers=1,
                        attention='dot',
                        clip_grad_value=0,
                        lexicon_loss_weight=0.025)

# deinfe save path
save_path = f'model/{experiment_no}'
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

tokenizer.get_all_tokens(all_sentence)
embedding.load_embedding(f'data/embedding/{config.embedding_file}',
                         tokenizer.all_tokens)

# build vocabulary dictionary
tokenizer.build_dict(all_sentence, embedding.word_dict)

# content encode to id
print('content encode --')
stance_data_df['target_encode'] = \
    tokenizer.encode(stance_data_df['target'].tolist())
stance_data_df['claim_encode'] = \
    tokenizer.encode(stance_data_df['claim'].tolist())
nli_data_df['target_encode'] = \
    tokenizer.encode(nli_data_df['target'].tolist())
nli_data_df['claim_encode'] = \
    tokenizer.encode(nli_data_df['claim'].tolist())

# label encode
print('label encode --')
if 'semeval' in config.stance_dataset:
    stance_label = {'favor': 0, 'against': 1, 'none': 2}
elif 'fnc' in config.stance_dataset:
    stance_label = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
nli_label = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

stance_data_df['label_encode'] = stance_data_df['label'].apply(
    lambda label: stance_label[label])
nli_data_df['label_encode'] = nli_data_df['label'].apply(
    lambda label: nli_label[label])

# load lexicon
lexicon = datas.load_lexicon(lexicon='emolex_emotion')

# content encode to lexicon vector
print('lexicon encode --')
stance_data_df['claim_lexicon'] = \
    tokenizer.encode_to_lexicon(stance_data_df['claim'].tolist(), lexicon)
nli_data_df['claim_lexicon'] = \
    tokenizer.encode_to_lexicon(nli_data_df['claim'].tolist(), lexicon)

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
        for task_id, x1, x2, lexicon, y in batch_iterator:
            # fed data into model
            x1 = x1.to(device)
            x2 = x2.to(device)
            pred_y, attn_weight = model(task_id, x1, x2)

            # evaluate loss
            batch_loss = loss.loss_function(lexicon_vector=lexicon,
                                            tokenizer=tokenizer, predict=pred_y,
                                            target=y, attn_weight=attn_weight,
                                            beta=config.lexicon_loss_weight,
                                            device=device)
            total_loss += batch_loss

            all_label_y.extend(y.tolist())
            all_pred_y.extend(torch.argmax(pred_y, axis=1).cpu().tolist())

    # evaluate loss and f1
    total_loss = total_loss / len(batch_iterator)
    f1 = sklearn.metrics.f1_score(all_label_y, all_pred_y, average='macro')

    if phase == 'train':
        return total_loss
    else:
        return total_loss, f1

# initialize loss and f1
all_train_loss = []
all_stance_loss, all_nli_loss = [], []
all_stance_f1, all_nli_f1 = [], []
best_train_loss, best_stance_loss, best_stance_f1 = None, None, None
best_fold, best_epoch = None, None

# define KFold
kf = KFold(n_splits=config.kfold, random_state=config.random_seed,
           shuffle=True)
stance_kf = kf.split(stance_data_df)
nli_kf = kf.split(nli_data_df)

# training
for fold, ((stance_train_index, stance_valid_index), \
    (nli_train_index, nli_valid_index)) in \
    enumerate(zip(stance_kf, nli_kf), start=1):
    # DataFrame
    stance_train_df = stance_data_df.iloc[stance_train_index]
    stance_valid_df = stance_data_df.iloc[stance_valid_index]
    nli_train_df = nli_data_df.iloc[nli_train_index]
    nli_valid_df = nli_data_df.iloc[nli_valid_index]

    # SingleTask Dataset
    stance_train_dataset = datas.SingleTaskDataset(
        task_id=0,
        target_encode=stance_train_df['target_encode'],
        claim_encode=stance_train_df['claim_encode'],
        claim_lexicon=stance_train_df['claim_lexicon'],
        label_encode=stance_train_df['label_encode'])
    stance_valid_dataset = datas.SingleTaskDataset(
        task_id=0,
        target_encode=stance_valid_df['target_encode'],
        claim_encode=stance_valid_df['claim_encode'],
        claim_lexicon=stance_valid_df['claim_lexicon'],
        label_encode=stance_valid_df['label_encode'])
    nli_train_dataset = datas.SingleTaskDataset(
        task_id=1,
        target_encode=nli_train_df['target_encode'],
        claim_encode=nli_train_df['claim_encode'],
        claim_lexicon=nli_train_df['claim_lexicon'],
        label_encode=nli_train_df['label_encode'])
    nli_valid_dataset = datas.SingleTaskDataset(
        task_id=1,
        target_encode=nli_valid_df['target_encode'],
        claim_encode=nli_valid_df['claim_encode'],
        claim_lexicon=nli_valid_df['claim_lexicon'],
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

    # construct model and optimizer
    model = models.Model(config=config,
                         num_embeddings=embedding.get_num_embeddings(),
                         padding_idx=tokenizer.pad_token_id,
                         embedding_weight=embedding.vector)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)

    # initialize loss and f1
    fold_train_loss = []
    fold_stance_loss, fold_nli_loss = [], []
    fold_stance_f1, fold_nli_f1 = [], []

    # train model
    model.zero_grad()

    for epoch in range(int(config.epoch)):
        model.train()
        print(f'{fold}-fold\n')
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc=f'epoch {epoch}, loss: {0:.3f}', position=0)

        for task_id, train_x1, train_x2, \
            train_lexicon, train_y in train_iterator:
            # fed data in into model
            train_x1 = train_x1.to(device)
            train_x2 = train_x2.to(device)
            pred_y, attn_weight = model(task_id, train_x1, train_x2)

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

            # update description
            train_iterator.set_description(f'epoch {epoch}, loss: {batch_loss:.3f}')

        # evaluate model
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc='evaluate training', position=0)
        stance_iterator = tqdm(stance_valid_dataloader,
                               total=len(stance_valid_dataloader),
                               desc='evaluate valid for stance', position=0)
        # nli_iterator = tqdm(nli_valid_dataloader,
        #                     total=len(nli_valid_dataloader),
        #                     desc='evaluate valid for nli', position=0)

        train_loss = evaluate(model, train_iterator, 'train')
        stance_loss, stance_f1 = evaluate(model, stance_iterator, 'test')
        # nli_loss, nli_f1 = evaluate(model, nli_iterator, 'test')

        print(f'train loss: {train_loss}, '
              f'stance loss: {stance_loss}, stance f1: {stance_f1}\n')

        # save model
        if best_stance_loss is None or stance_loss < best_stance_loss:
            # check model save path
            if not os.path.exists(f'{save_path}/{fold}-fold'):
                os.makedirs(f'{save_path}/{fold}-fold')
            torch.save(model.state_dict(),
                       f'{save_path}/{fold}-fold/model_{epoch}.ckpt')

            best_train_loss = train_loss
            best_stance_loss, best_stance_f1 = stance_loss, stance_f1
            best_fold, best_epoch = fold, epoch

        fold_train_loss.append(train_loss.item())
        fold_stance_loss.append(stance_loss.item())
        fold_stance_f1.append(stance_f1.item())
        # fold_nli_loss.append(nli_loss)
        # fold_nli_f1.append(nli_f1)

    all_train_loss.append(fold_train_loss)
    all_stance_loss.append(fold_stance_loss)
    all_stance_f1.append(fold_stance_f1)
    # all_nli_loss.append(fold_nli_loss)
    # all_nli_f1.append(fold_nli_f1)

# print final result
print(f'\n{best_fold}-fold epoch {best_epoch} - best train loss: {best_train_loss}, '
      f'best stance loss: {best_stance_loss}, best stance f1: {best_stance_f1}')

# init tensorboard
writer = SummaryWriter(f'tensorboard/experiment-{experiment_no}')

# write loss and f1 to tensorboard
for epoch in range(int(config.epoch)):
    writer.add_scalars(f'Loss/train',
                       {f'{fold}-fold': round(all_train_loss[fold][epoch], 3)
                       for fold in range(int(config.kfold))}, epoch)
    writer.add_scalars(f'Loss/valid_stance',
                       {f'{fold}-fold': round(all_stance_loss[fold][epoch], 3)
                       for fold in range(int(config.kfold))}, epoch)
    writer.add_scalars(f'F1/valid_stance',
                       {f'{fold}-fold': round(all_stance_f1[fold][epoch], 3)
                       for fold in range(int(config.kfold))}, epoch)
    # writer.add_scalars(f'Loss/valid_nli',
    #                    {f'{fold}-fold': round(all_nli_loss[fold][epoch], 3)
    #                    for fold in range(int(config.kfold))}, epoch)
    # writer.add_scalars(f'F1/valid_nli',
    #                    {f'{fold}-fold': all_nli_f1[fold][epoch]
    #                    for fold in range(int(config.kfold))}, epoch)

# add hyperparameters and final result to tensorboard
writer.add_hparams({
    'fold': str(best_fold),
    'epoch': str(best_epoch),
    'train_loss': best_train_loss.item(),
    'valid_stance_loss': best_stance_loss.item(),
    'valid_stance_f1': best_stance_f1.item(),
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