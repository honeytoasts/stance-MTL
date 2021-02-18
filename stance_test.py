# built-in module
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the stance-detection model'
    )

    # add argument to argparser
    parser.add_argument('--experiment_no',
                        default='1',
                        type=str)
    parser.add_argument('--epoch',
                        default='1',
                        type=str)

    return parser.parse_args()

def main():
    # pylint: disable=no-member

    # get experiment_no and epoch
    args = parse_args()
    experiment_no, epoch = args.experiment_no, args.epoch
    model_path = f'model/{experiment_no}'

    # load config, tokenizer, embedding
    config = util.config.load(f'{model_path}/config.json')

    if config.tokenizer == 'BaseTokenizer':
        tokenizer = util.tokenizer.BaseTokenizer(config)
    elif config.tokenizer == 'WordPunctTokenizer':
        tokenizer = util.tokenizer.WordPunctTokenizer(config)
    elif config.tokenizer == 'TweetTokenizer':
        tokenizer = util.tokenizer.TweetTokenizer(config)
    tokenizer.load(f'{model_path}/tokenizer.pickle')

    embedding = util.embedding.BaseEmbedding()
    embedding.load(f'{model_path}/embedding.pickle')

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load data
    data_df = util.data.load_dataset(dataset='semeval2016_test')

    # content encode
    data_df['target_encode'] = \
        tokenizer.encode(data_df['target'].tolist())
    data_df['claim_encode'] = \
        tokenizer.encode(data_df['claim'].tolist())

    # label encode
    if 'semeval' in config.stance_dataset:
        stance_label = {'favor': 0, 'against': 1, 'none': 2}
    elif 'fnc' in config.stance_dataset:
        stance_label = {'agree': 0, 'disagree': 1,
                        'discuss': 2, 'unrelated': 3}

    data_df['label_encode'] = data_df['label'].apply(
        lambda label: stance_label[label])

    # lexicon encode
    data_df['claim_lexicon'] = \
        tokenizer.encode_to_lexicon(data_df['claim_encode'].tolist())

    # construct dataset and dataloader
    dataset = util.data.SingleTaskDataset(
        task_id=0,
        target_name=data_df['target'],
        target_encode=data_df['target_encode'],
        claim_encode=data_df['claim_encode'],
        claim_lexicon=data_df['claim_lexicon'],
        label_encode=data_df['label_encode'])
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=util.data.SingleTaskDataset.collate_fn)

    # load model
    model = util.model.BaseModel(config=config,
                                 num_embeddings=embedding.get_num_embeddings(),
                                 padding_idx=tokenizer.pad_token_id,
                                 embedding_weight=embedding.vector)
    model.load_state_dict(
        torch.load(f'{model_path}/model_{epoch}.ckpt'))
    model = model.to(device)

    # evaluate
    batch_iterator = tqdm(dataloader, total=len(dataloader),
                          desc='evaluate test data', position=0)
    (loss, lexicon_loss, score) = \
        util.evaluate.stance_evaluate_function(device=device,
                                               model=model,
                                               config=config,
                                               batch_iterator=batch_iterator)

    # print loss and score
    if 'semeval' in config.stance_dataset:
        target_f1, macro_f1, micro_f1 = score

        print(f'loss: {round(loss, 5)}, '
              f'lexicon loss: {round(lexicon_loss, 5)}\n'
              f'macro f1: {round(macro_f1, 5)}, '
              f'micro f1: {round(micro_f1, 5)}\n'
              f'target f1: {target_f1}')
    elif 'fnc' in config.stance_dataset:
        print(f'loss: {round(loss, 5)}, '
              f'lexicon loss: {round(lexicon_loss, 5)}\n'
              f'weighted score: {round(score, 5)}, ')

    # initialize tensorboard
    writer = SummaryWriter(f'tensorboard/exp-{config.experiment_no}')

    # write loss and f1 to tensorboard
    if 'semeval' in config.stance_dataset:
        target_f1, macro_f1, micro_f1 = score

        writer.add_scalar('Loss/test_stance', loss, epoch)
        writer.add_scalar('Loss/test_lexicon', lexicon_loss, epoch)

        writer.add_scalar('F1/test_macro', macro_f1, epoch)
        writer.add_scalar('F1/test_micro', micro_f1, epoch)

        writer.add_scalars('F1/test_target',
                            {'atheism': target_f1[0],
                             'climate': target_f1[1],
                             'feminist': target_f1[2],
                             'hillary': target_f1[3],
                             'abortion': target_f1[4]}, epoch)
    elif 'fnc' in config.stance_dataset:
        writer.add_scalar('Loss/test_stance', loss, epoch)
        writer.add_scalar('Loss/test_lexicon', lexicon_loss, epoch)
        writer.add_scalar('F1/test_score', score, epoch)

    writer.close()

    # release GPU memory
    torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    main()