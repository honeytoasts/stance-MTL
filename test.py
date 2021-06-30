# built-in module
import argparse

# 3rd-party module
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# self-made module
import util

def get_config():
    parser = argparse.ArgumentParser(
        description='Evaluate the stance-detection model'
    )

    # add argument to argparser
    parser.add_argument('--experiment_no',
                        default='1',
                        type=str)
    parser.add_argument('--epoch',
                        default=-1,
                        type=int)
    parser.add_argument('--evaluate_nli',
                        default=0,
                        type=int)

    return parser.parse_args()

def test(experiment_no, epoch, evaluate_nli):
    # define model path
    model_path = f'data/model/{experiment_no}'

    # initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load config
    config_cls = util.config.BaseConfig()
    config_cls.load(file_path=f'{model_path}/config.json')
    config = config_cls.config

    # load tokenizer
    tokenizer = util.tokenizer.BaseTokenizer(config)
    tokenizer.load(file_path=f'{model_path}/tokenizer.pickle')

    # load embedding
    embedding = util.embedding.BaseEmbedding(
        embedding=config.embedding,
        embedding_dim=config.embedding_dim)
    embedding.load(file_path=f'{model_path}/embedding.pickle')

    # load data
    data = util.data.MultiData(training=False,
                               stance_target='all',
                               config=config,
                               tokenizer=tokenizer,
                               embedding=embedding)
    stance_data_df, nli_data_df = data.stance_train_df, data.nli_train_df
    stance_test_df, nli_test_df = data.stance_test_df, data.nli_test_df

    # split data to train and validation set
    stance_train_df, stance_valid_df = (
        train_test_split(stance_data_df,
                         test_size=float(config.test_size),
                         random_state=config.random_seed,
                         shuffle=True,
                         stratify=stance_data_df['label_encode']))
    nli_train_df, nli_valid_df = (
        train_test_split(nli_data_df,
                         test_size=float(config.test_size),
                         random_state=config.random_seed,
                         shuffle=True,
                         stratify=nli_data_df['label_encode']))

    # single-task dataset
    stance_train_dataset = util.data.SingleTaskDataset(
        data_id=stance_train_df['data_id'],
        task_id=0,
        target_name=stance_train_df['target'],
        task_target=stance_train_df['task_target_encode'],
        shared_target=stance_train_df['shared_target_encode'],
        task_claim=stance_train_df['task_claim_encode'],
        shared_claim=stance_train_df['shared_claim_encode'],
        task_attn_mask=stance_train_df['task_attn_mask'],
        shared_attn_mask=stance_train_df['shared_attn_mask'],
        labels=stance_train_df['label_encode'])
    stance_valid_dataset = util.data.SingleTaskDataset(
        data_id=stance_valid_df['data_id'],
        task_id=0,
        target_name=stance_valid_df['target'],
        task_target=stance_valid_df['task_target_encode'],
        shared_target=stance_valid_df['shared_target_encode'],
        task_claim=stance_valid_df['task_claim_encode'],
        shared_claim=stance_valid_df['shared_claim_encode'],
        task_attn_mask=stance_valid_df['task_attn_mask'],
        shared_attn_mask=stance_valid_df['shared_attn_mask'],
        labels=stance_valid_df['label_encode'])
    stance_test_dataset = util.data.SingleTaskDataset(
        data_id=stance_test_df['data_id'],
        task_id=0,
        target_name=stance_test_df['target'],
        task_target=stance_test_df['task_target_encode'],
        shared_target=stance_test_df['shared_target_encode'],
        task_claim=stance_test_df['task_claim_encode'],
        shared_claim=stance_test_df['shared_claim_encode'],
        task_attn_mask=stance_test_df['task_attn_mask'],
        shared_attn_mask=stance_test_df['shared_attn_mask'],
        labels=stance_test_df['label_encode'])

    nli_train_dataset = util.data.SingleTaskDataset(
        data_id=nli_train_df['data_id'],
        task_id=1,
        target_name=nli_train_df['target'],
        task_target=nli_train_df['task_target_encode'],
        shared_target=nli_train_df['shared_target_encode'],
        task_claim=nli_train_df['task_claim_encode'],
        shared_claim=nli_train_df['shared_claim_encode'],
        task_attn_mask=nli_train_df['task_attn_mask'],
        shared_attn_mask=nli_train_df['shared_attn_mask'],
        labels=nli_train_df['label_encode'])
    nli_valid_dataset = util.data.SingleTaskDataset(
        data_id=nli_valid_df['data_id'],
        task_id=1,
        target_name=nli_valid_df['target'],
        task_target=nli_valid_df['task_target_encode'],
        shared_target=nli_valid_df['shared_target_encode'],
        task_claim=nli_valid_df['task_claim_encode'],
        shared_claim=nli_valid_df['shared_claim_encode'],
        task_attn_mask=nli_valid_df['task_attn_mask'],
        shared_attn_mask=nli_valid_df['shared_attn_mask'],
        labels=nli_valid_df['label_encode'])
    nli_test_dataset = util.data.SingleTaskDataset(
        data_id=nli_test_df['data_id'],
        task_id=1,
        target_name=nli_test_df['target'],
        task_target=nli_test_df['task_target_encode'],
        shared_target=nli_test_df['shared_target_encode'],
        task_claim=nli_test_df['task_claim_encode'],
        shared_claim=nli_test_df['shared_claim_encode'],
        task_attn_mask=nli_test_df['task_attn_mask'],
        shared_attn_mask=nli_test_df['shared_attn_mask'],
        labels=nli_test_df['label_encode'])

    # single-task dataloader
    collate_fn = util.data.Collator(tokenizer.pad_token_id)

    stance_train_dataloader = DataLoader(
        dataset=stance_train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn)
    stance_valid_dataloader = DataLoader(
        dataset=stance_valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn)
    stance_test_dataloader = DataLoader(
        dataset=stance_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn)

    # multi-task dataset
    train_dataset = util.data.MultiTaskDataset(
        [stance_train_dataset, nli_train_dataset])
    valid_dataset = util.data.MultiTaskDataset(
        [stance_valid_dataset, nli_valid_dataset])
    test_dataset = util.data.MultiTaskDataset(
        [stance_test_dataset, nli_test_dataset])

    # multi-task batch sampler
    train_multitask_batch_sampler = util.data.MultiTaskBatchSampler(
        datasets=[stance_train_dataset, nli_train_dataset],
        batch_size=config.batch_size,
        random_seed=config.random_seed)
    valid_multitask_batch_sampler = util.data.MultiTaskBatchSampler(
        datasets=[stance_valid_dataset, nli_valid_dataset],
        batch_size=config.batch_size,
        random_seed=config.random_seed)
    test_multitask_batch_sampler = util.data.MultiTaskBatchSampler(
        datasets=[stance_test_dataset, nli_test_dataset],
        batch_size=config.batch_size,
        random_seed=config.random_seed)

    # multi-task dataloader
    collate_fn = util.data.Collator(tokenizer.pad_token_id)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_multitask_batch_sampler,
        collate_fn=collate_fn)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_sampler=valid_multitask_batch_sampler,
        collate_fn=collate_fn)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_sampler=test_multitask_batch_sampler,
        collate_fn=collate_fn)

    # load model
    if config.model == 'task-specific-shared':
        model = util.model.TaskSpecificSharedModel(
            config=config,
            num_embeddings=embedding.get_num_embeddings(),
            padding_idx=tokenizer.pad_token_id,
            embedding_weight=embedding.vector)
    elif config.model == 'shared':
        model = util.model.SharedModel(
            config=config,
            num_embeddings=embedding.get_num_embeddings(),
            padding_idx=tokenizer.pad_token_id,
            embedding_weight=embedding.vector)
    else: 
        raise ValueError(f'model {config.model} is not exist')

    model.load_state_dict(
        torch.load(f'{model_path}/model_{epoch}.ckpt'))
    model = model.to(device)

    # evaluate training data
    train_iterator = (
        tqdm(train_dataloader, total=len(train_dataloader),
             desc='evaluate training data', position=0)
        if evaluate_nli else
        tqdm(stance_train_dataloader, total=len(stance_train_dataloader),
             desc='evaluate training data', position=0))

    (train_total_loss, train_stance_loss, train_nli_loss,
     train_target_f1, train_macro_f1, train_micro_f1,
     train_nli_acc) = (
        util.evaluate.evaluate_function(device=device,
                                        model=model,
                                        config=config,
                                        batch_iterator=train_iterator,
                                        evaluate_nli=evaluate_nli))

    # evaluate validation data
    valid_iterator = (
        tqdm(valid_dataloader, total=len(valid_dataloader),
             desc='evaluate validation data', position=0)
        if evaluate_nli else
        tqdm(stance_valid_dataloader, total=len(stance_valid_dataloader),
             desc='evaluate validation data', position=0))

    (valid_total_loss, valid_stance_loss, valid_nli_loss,
     valid_target_f1, valid_macro_f1, valid_micro_f1,
     valid_nli_acc) = (
        util.evaluate.evaluate_function(device=device,
                                        model=model,
                                        config=config,
                                        batch_iterator=valid_iterator,
                                        evaluate_nli=evaluate_nli))

    # evaluate testing data
    test_iterator = (
        tqdm(test_dataloader, total=len(test_dataloader),
             desc='evaluate testing data', position=0)
        if evaluate_nli else
        tqdm(stance_test_dataloader, total=len(stance_test_dataloader),
             desc='evaluate testing data', position=0))

    (test_total_loss, test_stance_loss, test_nli_loss,
     test_target_f1, test_macro_f1, test_micro_f1,
     test_nli_acc) = (
        util.evaluate.evaluate_function(device=device,
                                        model=model,
                                        config=config,
                                        batch_iterator=test_iterator,
                                        evaluate_nli=evaluate_nli))

    # print loss
    print(f'train total loss: {round(train_total_loss, 5)}, '
          f'train stance loss: {round(train_stance_loss, 5)}, '
          f'train nli loss: {round(train_nli_loss, 5)}\n'
          f'valid total loss: {round(valid_total_loss, 5)}, '
          f'valid stance loss: {round(valid_stance_loss, 5)}, '
          f'valid nli loss: {round(valid_nli_loss, 5)}\n'
          f'test total loss: {round(test_total_loss, 5)}, '
          f'test stance loss: {round(test_stance_loss, 5)}, '
          f'test nli loss: {round(test_nli_loss, 5)}\n')

    # print target f1
    abbr_targets = ['atheism', 'climate', 'feminism', 'hillary', 'abortion']
    for i, abbr_target in enumerate(abbr_targets):
        print(f'Result for {abbr_target}:\n'
              f'train target f1: {round(train_target_f1[i], 5)}, '
              f'valid target f1: {round(valid_target_f1[i], 5)}, '
              f'test target f1: {round(test_target_f1[i], 5)}')

    # print f1 score
    print(f'\ntrain macro f1: {round(train_macro_f1, 5)}, '
          f'train micro f1: {round(train_micro_f1, 5)}, '
          f'train nli acc: {round(train_nli_acc, 5)}\n'
          f'valid macro f1: {round(valid_macro_f1, 5)}, '
          f'valid micro f1: {round(valid_micro_f1, 5)}, '
          f'valid nli acc: {round(valid_nli_acc, 5)}\n'
          f'test macro f1: {round(test_macro_f1, 5)}, '
          f'test micro f1: {round(test_micro_f1, 5)}, '
          f'test nli acc: {round(test_nli_acc, 5)}')

    return

def main():
    # get config
    config = get_config()
    
    # evaluate model
    test(experiment_no=config.experiment_no,
         epoch=config.epoch,
         evaluate_nli=config.evaluate_nli)

if __name__ == '__main__':
    main()