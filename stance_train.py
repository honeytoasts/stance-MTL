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

# prevent warning
pd.options.mode.chained_assignment = None

def main():
    # get config from command line
    config = util.config.parse_args()

    # define save path
    save_path = f'model/{config.experiment_no}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise ValueError(
            f'experiment {config.experiment_no} have already exist')

    # save config
    util.config.save(config, f'{save_path}/config.json')

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # set random seed and ensure deterministic
    os.environ['PYTHONHASHSEED'] = str(config.random_seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load data
    if config.stance_dataset == 'semeval2016':
        stance_data_df = util.data.load_dataset('semeval2016_train')
    elif config.stance_dataset == 'fnc-1':
        stance_data_df = util.data.load_dataset('fnc_train')

    if config.nli_dataset == 'mnli':
        nli_data_df = util.data.load_dataset('mnli_train')

    # initialize tokenizer
    if config.tokenizer == 'BaseTokenizer':
        tokenizer = util.tokenizer.BaseTokenizer(config)
    elif config.tokenizer == 'WordPunctTokenizer':
        tokenizer = util.tokenizer.WordPunctTokenizer(config)
    elif config.tokenizer == 'TweetTokenizer':
        tokenizer = util.tokenizer.TweetTokenizer(config)

    # initialize embedding
    embedding = util.embedding.BaseEmbedding(
        embedding_dim=config.embedding_dim)

    # get all tokens and embeddings
    all_sentence = []
    all_sentence.extend(stance_data_df['target'].drop_duplicates().tolist())
    all_sentence.extend(stance_data_df['claim'].drop_duplicates().tolist())
    all_sentence.extend(nli_data_df['target'].drop_duplicates().tolist())
    all_sentence.extend(nli_data_df['claim'].drop_duplicates().tolist())

    all_tokens = tokenizer.get_all_tokens(all_sentence)

    if 'cc.en.300.bin' in config.embedding:
        embedding.load_fasttext_embedding(tokens=all_tokens)
    else:
        embedding.load_file_embedding(
            embedding_path=f'data/embedding/{config.embedding}',
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
        stance_label = {'agree': 0, 'disagree': 1,
                        'discuss': 2, 'unrelated': 3}

    nli_label = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    stance_data_df['label_encode'] = stance_data_df['label'].apply(
        lambda label: stance_label[label])
    nli_data_df['label_encode'] = nli_data_df['label'].apply(
        lambda label: nli_label[label])

    # load lexicon
    lexicons = util.data.load_lexicon(lexicon=config.lexicon)

    # build lexicon dictionary
    tokenizer.build_lexicon_dict(lexicons)

    # lexicon encode
    stance_data_df['claim_lexicon'] = \
        tokenizer.encode_to_lexicon(stance_data_df['claim_encode'].tolist())
    nli_data_df['claim_lexicon'] = \
        tokenizer.encode_to_lexicon(nli_data_df['claim_encode'].tolist())

    # save tokenizer and embedding
    tokenizer.save(f'{save_path}/tokenizer.pickle')
    embedding.save(f'{save_path}/embedding.pickle')

    # initialize tensorboard
    writer = SummaryWriter(f'tensorboard/exp-{config.experiment_no}')

    # initialize loss and f1 score
    best_train_total_loss = None
    best_train_stance_loss, best_valid_stance_loss = None, None
    best_train_stance_f1, best_valid_stance_f1 = None, None
    best_epoch = None

    # split data to train and validation set
    stance_train_df, stance_valid_df = \
        train_test_split(stance_data_df,
                        test_size=float(config.test_size),
                        random_state=config.random_seed)
    nli_train_df, nli_valid_df = \
        train_test_split(nli_data_df,
                        test_size=float(config.test_size),
                        random_state=config.random_seed)

    # single-task dataset
    stance_train_dataset = util.data.SingleTaskDataset(
        task_id=0,
        target_name=stance_train_df['target'],
        target_encode=stance_train_df['target_encode'],
        claim_encode=stance_train_df['claim_encode'],
        claim_lexicon=stance_train_df['claim_lexicon'],
        label_encode=stance_train_df['label_encode'])
    stance_valid_dataset = util.data.SingleTaskDataset(
        task_id=0,
        target_name=stance_valid_df['target'],
        target_encode=stance_valid_df['target_encode'],
        claim_encode=stance_valid_df['claim_encode'],
        claim_lexicon=stance_valid_df['claim_lexicon'],
        label_encode=stance_valid_df['label_encode'])
    nli_train_dataset = util.data.SingleTaskDataset(
        task_id=1,
        target_name=nli_train_df['target'],
        target_encode=nli_train_df['target_encode'],
        claim_encode=nli_train_df['claim_encode'],
        claim_lexicon=nli_train_df['claim_lexicon'],
        label_encode=nli_train_df['label_encode'])
    nli_valid_dataset = util.data.SingleTaskDataset(
        task_id=1,
        target_name=nli_valid_df['target'],
        target_encode=nli_valid_df['target_encode'],
        claim_encode=nli_valid_df['claim_encode'],
        claim_lexicon=nli_valid_df['claim_lexicon'],
        label_encode=nli_valid_df['label_encode'])

    # multi-task dataset
    train_dataset = util.data.MultiTaskDataset(
        [stance_train_dataset, nli_train_dataset])
    valid_dataset = util.data.MultiTaskDataset(
        [stance_valid_dataset, nli_valid_dataset])

    # multi-task batch sampler
    train_multitask_batch_sampler = util.data.MultiTaskBatchSampler(
        datasets=[stance_train_dataset, nli_train_dataset],
        batch_size=config.batch_size,
        random_seed=config.random_seed)
    valid_multitask_batch_sampler = util.data.MultiTaskBatchSampler(
        datasets=[stance_valid_dataset, nli_valid_dataset],
        batch_size=config.batch_size,
        random_seed=config.random_seed)

    # multi-task dataLoader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_multitask_batch_sampler,
        collate_fn=util.data.SingleTaskDataset.collate_fn)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_sampler=valid_multitask_batch_sampler,
        collate_fn=util.data.SingleTaskDataset.collate_fn)

    # construct model, optimizer and scheduler
    model = util.model.BaseModel(config=config,
                                num_embeddings=embedding.get_num_embeddings(),
                                padding_idx=tokenizer.pad_token_id,
                                embedding_weight=embedding.vector)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    if float(config.lr_decay) != 1:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay)

    # training model
    model.zero_grad()

    for epoch in range(int(config.epoch)):
        print('\n')
        model.train()
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc=f'epoch {epoch}', position=0)

        for task_id, _, train_x1, train_x2, \
            train_lexicon, train_y in train_iterator:
            # specify device for data
            train_x1 = train_x1.to(device)
            train_x2 = train_x2.to(device)
            train_lexicon = train_lexicon.to(device)
            train_y = train_y.to(device)

            # get predict label and attention weight
            pred_y, task_weight, shared_weight = \
                model(task_id, train_x1, train_x2)

            # clean up gradient
            optimizer.zero_grad()

            # calculate loss
            batch_loss, _ = util.loss.loss_function(
                task_id=task_id,
                predict=pred_y,
                target=train_y,
                lexicon_vector=train_lexicon,
                task_weight=task_weight,
                shared_weight=shared_weight,
                nli_loss_weight=config.nli_loss_weight,
                lexicon_loss_weight=config.lexicon_loss_weight)

            # backward pass
            batch_loss.backward(retain_graph=True)

            # prevent gradient boosting or vanishing
            if config.clip_grad_value != 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                                                config.clip_grad_value)

            # gradient decent
            optimizer.step()

            # apply scheduler
            if float(config.lr_decay) != 1:
                scheduler.step()

        # evaluate model
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc='evaluate training data', position=0)
        (train_total_loss, train_total_lexicon_loss,
         train_stance_loss, train_stance_lexicon_loss,
         train_nli_loss, train_nli_lexicon_loss,
         train_stance_score, train_nli_score) = \
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=train_iterator)

        valid_iterator = tqdm(valid_dataloader, total=len(valid_dataloader),
                              desc='evaluate validation data', position=0)
        (_, _,
         valid_stance_loss, valid_stance_lexicon_loss,
         valid_nli_loss, valid_nli_lexicon_loss,
         valid_stance_score, valid_nli_score) = \
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=valid_iterator)

        # get stance score
        if 'semeval' in config.stance_dataset:
            train_stance_f1 = train_stance_score[2]  # micro-f1 score
            valid_stance_f1 = valid_stance_score[2]
        elif 'fnc' in config.stance_dataset:
            train_stance_f1 = train_stance_score  # weighted score
            valid_stance_f1 = valid_stance_score

        # print loss and score
        print(f'train total loss : {round(train_total_loss, 5)}, '
              f'train total lexicon loss : {round(train_total_lexicon_loss, 5)}\n'
              f'train stance loss: {round(train_stance_loss, 5)}, '
              f'train stance lexicon loss: {round(train_stance_lexicon_loss, 5)}\n'
              f'valid stance loss: {round(valid_stance_loss, 5)}, '
              f'valid stance lexicon loss: {round(valid_stance_lexicon_loss, 5)}\n'
              f'train stance f1: {round(train_stance_f1, 5)}, '
              f'valid stance f1: {round(valid_stance_f1, 5)}')

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
        if 'semeval' in config.stance_dataset:
            train_target_f1, train_macro_f1, train_micro_f1 = \
                train_stance_score
            valid_target_f1, valid_macro_f1, valid_micro_f1 = \
                valid_stance_score

            writer.add_scalar('F1/train_macro', train_macro_f1, epoch)
            writer.add_scalar('F1/train_micro', train_micro_f1, epoch)
            writer.add_scalar('F1/valid_macro', valid_macro_f1, epoch)
            writer.add_scalar('F1/valid_micro', valid_micro_f1, epoch)

            writer.add_scalars('F1/train_target',
                               {'atheism': train_target_f1[0],
                                'climate': train_target_f1[1],
                                'feminist': train_target_f1[2],
                                'hillary': train_target_f1[3],
                                'abortion': train_target_f1[4]}, epoch)
            writer.add_scalars('F1/valid_target',
                               {'atheism': valid_target_f1[0],
                                'climate': valid_target_f1[1],
                                'feminist': valid_target_f1[2],
                                'hillary': valid_target_f1[3],
                                'abortion': valid_target_f1[4]}, epoch)
        elif 'fnc' in config.stance_dataset:
            writer.add_scalar('F1/train_score', train_stance_score, epoch)
            writer.add_scalar('F1/valid_score', valid_stance_score, epoch)

        writer.add_scalar('F1/train_nli', train_nli_score, epoch)
        writer.add_scalar('F1/valid_nli', valid_nli_score, epoch)

    # print final result
    print(f'\nexperiment {config.experiment_no}: epoch {best_epoch}\n'
          f'best train total loss : {best_train_total_loss}\n'
          f'best train stance loss: {best_train_stance_loss}, '
          f'best valid stance loss: {best_valid_stance_loss}\n'
          f'best train stance f1  : {best_train_stance_f1}, '
          f'best valid stance f1  : {best_valid_stance_f1}')

    # add hyperparameters and final result to tensorboard
    writer.add_hparams({
        'epoch': best_epoch,
        'train_total_loss': best_train_total_loss,
        'train_stance_loss': best_train_stance_loss,
        'valid_stance_loss': best_valid_stance_loss,
        'train_stance_f1': best_train_stance_f1,
        'valid_stance_f1': best_valid_stance_f1
    }, metric_dict={})
    writer.add_hparams(
        {key: str(value) for key, value in config.__dict__.items()},
        metric_dict={})
    writer.close()

    # release GPU memory
    torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    main()