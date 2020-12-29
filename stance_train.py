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
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorboard

# self-made module
import util

# hyperparameter setting
experiment_no = 61
config = util.config.BaseConfig(# experiment no
                                experiment_no=experiment_no,
                                # preprocess
                                tokenizer='WordPunctTokenizer',
                                filter='all',
                                min_count=1,
                                max_seq_len=20,
                                # dataset and lexicon
                                stance_dataset='semeval2016',
                                embedding_file='fasttext/crawl-300d-2M.vec',
                                lexicon_file='emolex_emotion',
                                # hyperparameter
                                embedding_dim=300,
                                task_hidden_dim=100,
                                shared_hidden_dim=100,
                                num_rnn_layers=1,
                                num_linear_layers=1,
                                attention='dot',
                                dropout=0.2,
                                learning_rate=5e-5,
                                clip_grad_value=0,
                                weight_decay=0,
                                lr_decay_step=10,
                                lr_decay=1,
                                nli_loss_weight=1.0,
                                lexicon_loss_weight=0,
                                random_seed=77,
                                kfold=5,
                                train_test_split=0.2,
                                epoch=70,
                                batch_size=32)

# define save path
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
    # torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# load data
if config.stance_dataset == 'semeval2016':
    stance_data_df = util.data.load_dataset('semeval2016_train')
elif config.stance_dataset == 'fnc-1':
    stance_data_df = util.data.load_dataset('fnc_train')

nli_data_df = util.data.load_dataset('mnli_train')

# initialize tokenizer
if config.tokenizer == 'BaseTokenizer':
    tokenizer = util.tokenizer.BaseTokenizer(config)
elif config.tokenizer == 'WordPunctTokenizer':
    tokenizer = util.tokenizer.WordPunctTokenizer(config)

# initialize embedding
if any([embedding in config.embedding_file for embedding in ['glove', 'fasttext']]):
    embedding = util.embedding.BaseEmbedding(embedding_dim=config.embedding_dim)

# get all tokens and embeddings
all_sentence = []
all_sentence.extend(stance_data_df['target'].drop_duplicates().tolist())
all_sentence.extend(stance_data_df['claim'].drop_duplicates().tolist())
all_sentence.extend(nli_data_df['target'].drop_duplicates().tolist())
all_sentence.extend(nli_data_df['claim'].drop_duplicates().tolist())

all_tokens = tokenizer.get_all_tokens(all_sentence)
embedding.load_embedding(embedding_path=f'data/embedding/{config.embedding_file}',
                         tokens=all_tokens)

# build vocabulary dictionary
tokenizer.build_dict(embedding.word_dict)

# encode content to id
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

# load lexicon
lexicons = util.data.load_lexicon(lexicon=config.lexicon_file)

# build lexicon dictionary
tokenizer.build_lexicon_dict(lexicons)

# encode content to lexicon vector
stance_data_df['claim_lexicon'] = \
    tokenizer.encode_to_lexicon(stance_data_df['claim_encode'].tolist())
nli_data_df['claim_lexicon'] = \
    tokenizer.encode_to_lexicon(nli_data_df['claim_encode'].tolist())

# save config, tokenizer and embedding
config_path = f'{save_path}/config.json'
tokenizer_path = f'{save_path}/tokenizer.pickle'
embedding_path = f'{save_path}/embedding.pickle'

config.save(config_path)
tokenizer.save(tokenizer_path)
embedding.save(embedding_path)

# initialize loss and f1
best_train_total_loss = None
best_train_stance_loss, best_valid_stance_loss = None, None
best_train_stance_f1, best_valid_stance_f1 = None, None
best_epoch = None

# initialize tensorboard
writer = SummaryWriter(f'tensorboard/exp-{experiment_no}')

# split data to train and valid set
stance_train_df, stance_valid_df = \
    train_test_split(stance_data_df,
                     test_size=float(config.train_test_split),
                     random_state=config.random_seed)
nli_train_df, nli_valid_df = \
    train_test_split(nli_data_df,
                     test_size=float(config.train_test_split),
                     random_state=config.random_seed)

# SingleTask Dataset
stance_train_dataset = util.data.SingleTaskDataset(
    task_id=0,
    target_encode=stance_train_df['target_encode'],
    claim_encode=stance_train_df['claim_encode'],
    claim_lexicon=stance_train_df['claim_lexicon'],
    label_encode=stance_train_df['label_encode'])
stance_valid_dataset = util.data.SingleTaskDataset(
    task_id=0,
    target_encode=stance_valid_df['target_encode'],
    claim_encode=stance_valid_df['claim_encode'],
    claim_lexicon=stance_valid_df['claim_lexicon'],
    label_encode=stance_valid_df['label_encode'])
nli_train_dataset = util.data.SingleTaskDataset(
    task_id=1,
    target_encode=nli_train_df['target_encode'],
    claim_encode=nli_train_df['claim_encode'],
    claim_lexicon=nli_train_df['claim_lexicon'],
    label_encode=nli_train_df['label_encode'])
nli_valid_dataset = util.data.SingleTaskDataset(
    task_id=1,
    target_encode=nli_valid_df['target_encode'],
    claim_encode=nli_valid_df['claim_encode'],
    claim_lexicon=nli_valid_df['claim_lexicon'],
    label_encode=nli_valid_df['label_encode'])

# MultiTask Dataset
train_dataset = util.data.MultiTaskDataset(
    [stance_train_dataset, nli_train_dataset])

# SingleTask Dataloader
stance_valid_dataloader = DataLoader(
    dataset=stance_valid_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=util.data.SingleTaskDataset.collate_fn)
nli_valid_dataloader = DataLoader(
    dataset=nli_valid_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=util.data.SingleTaskDataset.collate_fn)

# MultiTask DataLoader
multitask_batch_sampler = util.data.MultiTaskBatchSampler(
    datasets=[stance_train_dataset, nli_train_dataset],
    batch_size=config.batch_size,
    random_seed=config.random_seed)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_sampler=multitask_batch_sampler,
    collate_fn=util.data.SingleTaskDataset.collate_fn)

# construct model, optimizer and scheduler
model = util.model.BaseModel(config=config,
                             num_embeddings=embedding.get_num_embeddings(),
                             padding_idx=tokenizer.pad_token_id,
                             embedding_weight=embedding.vector)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.learning_rate,
                             weight_decay=config.weight_decay)

if float(config.lr_decay) != 1:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=config.lr_decay_step,
                                                gamma=config.lr_decay)

# training model
model.zero_grad()

for epoch in range(int(config.epoch)):
    print('\n')
    model.train()
    train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                          desc=f'epoch {epoch}', position=0)

    for task_id, train_x1, train_x2, \
        train_lexicon, train_y in train_iterator:
        # specify device for data
        train_x1 = train_x1.to(device)
        train_x2 = train_x2.to(device)
        train_lexicon = train_lexicon.to(device)
        train_y = train_y.to(device)

        # get predict label and attention weight
        pred_y, attn_weight = model(task_id, train_x1, train_x2)

        # clean up gradient
        optimizer.zero_grad()

        # calculate loss
        batch_loss, _ = \
            util.loss.loss_function(task_id=task_id,
                                    lexicon_vector=train_lexicon,
                                    predict=pred_y,
                                    target=train_y,
                                    attn_weight=attn_weight,
                                    nli_loss_weight=config.nli_loss_weight,
                                    lexicon_loss_weight=config.lexicon_loss_weight)

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
                          desc='evaluate training', position=0)
    (train_total_loss, train_total_lexicon_loss,
     train_stance_loss, train_stance_lexicon_loss,
     train_nli_loss, train_nli_lexicon_loss,
     train_stance_f1) = \
        util.evaluate.evaluate_function(model=model,
                                        config=config,
                                        batch_iterator=train_iterator,
                                        device=device,
                                        phase='train')

    stance_iterator = tqdm(stance_valid_dataloader,
                           total=len(stance_valid_dataloader),
                           desc='evaluate valid for stance', position=0)
    valid_stance_loss, valid_stance_lexicon_loss, valid_stance_f1= \
        util.evaluate.evaluate_function(model=model,
                                        config=config,
                                        batch_iterator=stance_iterator,
                                        device=device,
                                        phase='valid')

    nli_iterator = tqdm(nli_valid_dataloader,
                        total=len(nli_valid_dataloader),
                        desc='evaluate valid for nli', position=0)
    valid_nli_loss, valid_nli_lexicon_loss, valid_nli_f1= \
        util.evaluate.evaluate_function(model=model,
                                        config=config,
                                        batch_iterator=nli_iterator,
                                        device=device,
                                        phase='valid')

    print(f'train total loss : {round(train_total_loss.item(), 5)}, '
          f'train total lexicon loss : {round(train_total_lexicon_loss.item(), 5)}\n'
          f'train stance loss: {round(train_stance_loss.item(), 5)}, '
          f'train stance lexicon loss: {round(train_stance_lexicon_loss.item(), 5)}\n'
          f'valid stance loss: {round(valid_stance_loss.item(), 5)}, '
          f'valid stance lexicon loss: {round(valid_stance_lexicon_loss.item(), 5)}\n'
          f'train stance f1: {round(train_stance_f1.item(), 5)}, '
          f'valid stance f1: {round(valid_stance_f1.item(), 5)}')

    # save model
    if (best_valid_stance_loss is None) or \
        (valid_stance_loss < best_valid_stance_loss) or \
        (valid_stance_f1 > best_valid_stance_f1):

        torch.save(model.state_dict(), f'{save_path}/model_{epoch}.ckpt')

        best_train_total_loss = train_total_loss
        best_train_stance_loss = train_stance_loss
        best_valid_stance_loss = valid_stance_loss
        best_train_stance_f1 = train_stance_f1
        best_valid_stance_f1 = valid_stance_f1
        best_epoch = epoch

    # apply scheduler
    if float(config.lr_decay) != 1:
        scheduler.step()

    # write loss to tensorboard
    writer.add_scalar('Loss/train_total',  train_total_loss, epoch)
    writer.add_scalar('Loss/train_total_lexicon', train_total_lexicon_loss, epoch)
    writer.add_scalar('Loss/train_stance', train_stance_loss, epoch)
    writer.add_scalar('Loss/train_stance_lexicon', train_stance_lexicon_loss, epoch)
    writer.add_scalar('Loss/train_nli', train_nli_loss, epoch)
    writer.add_scalar('Loss/train_nli_lexicon', train_nli_lexicon_loss, epoch)

    writer.add_scalar('Loss/valid_stance', valid_stance_loss, epoch)
    writer.add_scalar('Loss/valid_stance_lexicon', valid_stance_lexicon_loss, epoch)

    writer.add_scalar('Loss/valid_nli', valid_nli_loss, epoch)
    writer.add_scalar('Loss/valid_nli_lexicon', valid_nli_lexicon_loss, epoch)

    # write f1 to tensorboard
    writer.add_scalar('F1/train_stance', train_stance_f1, epoch)
    writer.add_scalar('F1/valid_stance', valid_stance_f1, epoch)


# print final result
print(f'\nexperiment {experiment_no}: epoch {best_epoch}\n'
      f'best train total loss : {best_train_total_loss}\n'
      f'best train stance loss: {best_train_stance_loss}, '
      f'best valid stance loss: {best_valid_stance_loss}\n'
      f'best train stance f1  : {best_train_stance_f1}, '
      f'best valid stance f1  : {best_valid_stance_f1}')

# add hyperparameters and final result to tensorboard
writer.add_hparams({
    # experiment result
    'epoch': str(best_epoch),
    'train_total_loss': best_train_total_loss.item(),
    'train_stance_loss': best_train_stance_loss.item(),
    'valid_stance_loss': best_valid_stance_loss.item(),
    'train_stance_f1': best_train_stance_f1.item(),
    'valid_stance_f1': best_valid_stance_f1.item(),
    # preprocess
    'tokenizer': str(config.tokenizer),
    'filter': str(config.filter),
    'min_count': str(config.min_count),
    'max_seq_len': str(config.max_seq_len),
    # dataset and lexicon
    'stance_dataset': config.stance_dataset,
    'embedding_file': config.embedding_file,
    'lexicon_file': config.lexicon_file,
    # hyperparameter
    'embedding_dim': str(config.embedding_dim),
    'task_hidden_dim': str(config.task_hidden_dim),
    'shared_hidden_dim': str(config.shared_hidden_dim),
    'num_rnn_layers': str(config.num_rnn_layers),
    'num_linear_layers': str(config.num_linear_layers),
    'attention': str(config.attention),
    'dropout': str(config.dropout),
    'lr': str(config.learning_rate),
    'clip_grad_value': str(config.clip_grad_value),
    'weight_decay': str(config.weight_decay),
    'lr_decay_step': str(config.lr_decay_step),
    'lr_decay': str(config.lr_decay),
    'nli_loss_weight': str(config.nli_loss_weight),
    'lexicon_loss_weight': str(config.lexicon_loss_weight),
    'batch_size': str(config.batch_size),
}, metric_dict={})
writer.close()

# release GPU memory
torch.cuda.empty_cache()
