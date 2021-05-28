# built-in module
import os
import random

# 3rd-party module
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# self-made module
import util

# prevent warning
pd.options.mode.chained_assignment = None

def main():
    # get config
    config_cls = util.config.BaseConfig()
    config_cls.get_config()
    config = config_cls.config

    # define save path
    save_path = f'model/{config.experiment_no}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise FileExistsError(f'experiment {config.experiment_no} is exist')

    # save config
    config_cls.save(f'{save_path}/config.json')

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

    # initialize tokenizer
    tokenizer = util.tokenizer.BaseTokenizer(config)

    # initialize embedding
    embedding = util.embedding.BaseEmbedding(
        embedding=config.embedding,
        embedding_dim=config.embedding_dim)

    # load data
    data = util.data.MultiData(training=True,
                               stance_target='all',
                               config=config,
                               tokenizer=tokenizer,
                               embedding=embedding)
    stance_data_df, nli_data_df = data.stance_train_df, data.nli_train_df

    # split data to train and validation set
    stance_train_df, stance_valid_df = (
        train_test_split(stance_data_df,
                         test_size=float(config.test_size),
                         random_state=config.random_seed))
    nli_train_df, nli_valid_df = (
        train_test_split(nli_data_df,
                        test_size=float(config.test_size),
                        random_state=config.random_seed))

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

    # initialize f1 score
    best_valid_stance_f1 = None

    # save tokenizer and embedding
    tokenizer.save(f'{save_path}/tokenizer.pickle')
    embedding.save(f'{save_path}/embedding.pickle')

    # initialize tensorboard
    writer = SummaryWriter(f'tensorboard/{config.experiment_no}')

    # construct model
    if config.model == 'task-specific-shared':
        model = util.model.TaskSpecificSharedModel(
            config=config,
            num_embeddings=embedding.get_num_embeddings(),
            padding_idx=tokenizer.pad_token_id,
            embedding_weight=embedding.vector)
    else: 
        raise ValueError(f'model {config.model} is not exist')
    model = model.to(device)

    # construct optimizer
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    # construct scheduler
    if float(config.lr_decay) != 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.lr_decay_step,
            gamma=config.lr_decay)

    # training model
    model.zero_grad()

    for epoch in range(int(config.epoch)):
        print()
        model.train()
        train_iterator = tqdm(train_dataloader, total=len(train_dataloader),
                              desc=f'epoch {epoch}', position=0)

        for (task_id, _,
             task_target, shared_target,
             task_claim, shared_claim,
             task_attn_mask, shared_attn_mask,
             task_adj_matrix, shared_adj_matrix,
             label) in train_iterator:

            # specify device for data
            task_target = task_target.to(device)
            shared_target = shared_target.to(device)
            task_claim = task_claim.to(device)
            shared_claim = shared_claim.to(device)
            task_attn_mask = task_attn_mask.to(device)
            shared_attn_mask = shared_attn_mask.to(device)
            task_adj_matrix = task_adj_matrix.to(device)
            shared_adj_matrix = shared_adj_matrix.to(device)
            label = label.to(device)

            # get predict label
            predict, _ = model(task_id,
                            task_target, shared_target,
                            task_claim, shared_claim,
                            task_attn_mask, shared_attn_mask,
                            task_adj_matrix, shared_adj_matrix)

            # calculate loss
            batch_loss = (
                util.loss.loss_function(
                    task_id=task_id,
                    predict=predict,
                    target=label,
                    nli_loss_weight=config.nli_loss_weight))

            # backward pass
            batch_loss.backward(retain_graph=True)

            # prevent gradient boosting or vanishing
            if config.clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               config.clip_grad_value)

            # gradient decent
            optimizer.step()

            # apply scheduler
            if float(config.lr_decay) != 1.0:
                scheduler.step()

        # evaluate model
        train_iterator = (
            tqdm(train_dataloader, total=len(train_dataloader),
                 desc='evaluate training data', position=0)
            if config.evaluate_nli else
            tqdm(stance_train_dataloader, total=len(stance_train_dataloader),
                 desc='evaluate training data', position=0))

        (train_total_loss, train_stance_loss, train_nli_loss,
         train_target_f1, train_macro_f1, train_micro_f1,
         train_nli_acc) = (
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=train_iterator,
                                            evaluate_nli=config.evaluate_nli))

        valid_iterator = (
            tqdm(valid_dataloader, total=len(valid_dataloader),
                 desc='evaluate validation data', position=0)
            if config.evaluate_nli else
            tqdm(stance_valid_dataloader, total=len(stance_valid_dataloader),
                 desc='evaluate validation data', position=0))

        (valid_total_loss, valid_stance_loss, valid_nli_loss,
         valid_target_f1, valid_macro_f1, valid_micro_f1,
         valid_nli_acc) = (
            util.evaluate.evaluate_function(device=device,
                                            model=model,
                                            config=config,
                                            batch_iterator=valid_iterator,
                                            evaluate_nli=config.evaluate_nli))

        # print loss and score
        print(f'train total loss : {round(train_total_loss, 5)}, '
              f'train stance loss: {round(train_stance_loss, 5)}, '
              f'valid stance loss: {round(valid_stance_loss, 5)}, '
              f'train stance f1: {round(train_micro_f1, 5)}, '
              f'valid stance f1: {round(valid_micro_f1, 5)}')

        # save model
        if best_valid_stance_f1 is None or (
            valid_micro_f1 > best_valid_stance_f1):

            best_valid_stance_f1 = valid_micro_f1

            torch.save(model.state_dict(),
                       f'{save_path}/model_{epoch}.ckpt')

        # write loss to tensorboard
        writer.add_scalar('Loss/train_total',  train_total_loss, epoch)
        writer.add_scalar('Loss/train_stance', train_stance_loss, epoch)
        writer.add_scalar('Loss/train_nli', train_nli_loss, epoch)

        writer.add_scalar('Loss/valid_total',  valid_total_loss, epoch)
        writer.add_scalar('Loss/valid_stance', valid_stance_loss, epoch)
        writer.add_scalar('Loss/valid_nli', valid_nli_loss, epoch)

        # write f1 to tensorboard
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

        writer.add_scalar('F1/train_nli', train_nli_acc, epoch)
        writer.add_scalar('F1/valid_nli', valid_nli_acc, epoch)

    # print final result
    print(f'\nexperiment {config.experiment_no}:\n'
          f'best valid stance f1  : {best_valid_stance_f1}')

    # add hyperparameters to tensorboard
    writer.add_hparams(
        {key: str(value) for key, value in config.__dict__.items()},
        metric_dict={})
    writer.close()

    # release GPU memory
    torch.cuda.empty_cache()

    return

if __name__ == '__main__':
    main()