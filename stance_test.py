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
import util

# parameter and model path setting
experiment_no = 43
epoch = 19
model_path = f'model/{experiment_no}/'

# load config, tokenizer, embedding
config = util.config.BaseConfig()
config = config.load(f'{model_path}/config.json')

tokenizer = util.tokenizer.WordPunctTokenizer(config)
tokenizer.load(f'{model_path}/tokenizer.pickle')

embedding = util.embedding.BaseEmbedding()
embedding.load(f'{model_path}/embedding.pickle')

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
    data_df = util.data.load_dataset('semeval2016_test')
elif config.stance_dataset == 'fnc-1':
    data_df = util.data.load_dataset('fnc_test')

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
lexicons = util.data.load_lexicon(lexicon=config.lexicon_file)

# build lexicon dictionary
tokenizer.build_lexicon_dict(lexicons)

# encode content to lexicon vector
data_df['claim_lexicon'] = \
    tokenizer.encode_to_lexicon(data_df['claim_encode'].tolist())

# define dataset and dataloader
dataset = util.data.SingleTaskDataset(
    task_id=0,
    target_encode=data_df['target_encode'],
    claim_encode=data_df['claim_encode'],
    claim_lexicon=data_df['claim_lexicon'],
    label_encode=data_df['label_encode'])
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=util.data.SingleTaskDataset.collate_fn)

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
            lexicon = lexicon.to(device)
            y = y.to(device)
            pred_y, attn_weight = model(task_id, x1, x2)

            # evaluate loss
            batch_loss, batch_lexicon_loss = \
                util.loss.loss_function(task_id=task_id,
                                        lexicon_vector=lexicon,
                                        predict=pred_y,
                                        target=y,
                                        attn_weight=attn_weight,
                                        nli_loss_weight=1.0,
                                        lexicon_loss_weight=config.lexicon_loss_weight)
            total_loss += batch_loss
            total_lexicon_loss += batch_lexicon_loss

            all_label_y.extend(y.tolist())
            all_pred_y.extend(torch.argmax(pred_y, axis=1).cpu().tolist())

    # evaluate loss
    total_loss = total_loss / len(batch_iterator)
    total_lexicon_loss = total_lexicon_loss / len(batch_iterator)

    # get score and report
    targets = (data_df['target']
               if config.stance_dataset == 'semeval2016' else None)
    scores = util.scorer.score_function(dataset=config.stance_dataset,
                                        label_y=all_label_y,
                                        pred_y=all_pred_y,
                                        targets=targets)

    return total_loss, total_lexicon_loss, scores

# load model
model = util.model.BaseModel(config=config,
                             num_embeddings=embedding.get_num_embeddings(),
                             padding_idx=tokenizer.pad_token_id,
                             embedding_weight=embedding.vector)
model.load_state_dict(
    torch.load(f'{model_path}/model_{epoch}.ckpt'))
model = model.to(device)

# evaluate
model.eval()

test_iterator = tqdm(dataloader, total=len(dataloader),
                     desc='evaluate test set', position=0)
loss, lexicon_loss, scores = evaluate(model, test_iterator)

# print result
if config.stance_dataset == 'semeval2016':
    _, _, _, _, target_f1, macro_f1, micro_f1 = scores

    print(f'\nexperiment {experiment_no}: {epoch}-epoch\n'
          f'dataset: {config.stance_dataset}\n'
          f'loss: {round(loss.item(), 5)}, '
          f'lexicon loss: {round(lexicon_loss.item(), 5)}\n'
          f'target f1: {target_f1}\n'
          f'macro f1: {macro_f1}, micro f1: {micro_f1}\n')

elif config.stance_dataset == 'fnc-1':
    _, precision, recall, f1 = scores  # pylint: disable=unbalanced-tuple-unpacking

    print(f'\nexperiment {experiment_no}: {epoch}-epoch\n'
          f'dataset: {config.stance_dataset}\n'
          f'loss: {round(loss.item(), 5)}, '
          f'lexicon loss: {round(lexicon_loss.item(), 5)}\n'
          f'f1: {f1}\n')