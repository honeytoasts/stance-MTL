# built-in module
import unicodedata
import random

# 3rd-party module
import pandas as pd
from tqdm import tqdm
import torch

def preprocessing(data):
    # encoding normalize
    data = [[unicodedata.normalize('NFKC', str(column))
             for column in row] for row in data]

    # change to lowercase
    data = [[column.lower().strip() for column in row] for row in data]

    return data

def convert_to_dataframe(data):
    target = [row[0] for row in data]
    claim = [row[1] for row in data]
    label = [row[2] for row in data]

    data_df = pd.DataFrame({'target': target, 'claim': claim, 'label': label})

    return data_df

def load_dataset_semeval2016(split='train'):
    # file path
    if split == 'train':
        file_path = ('data/semeval2016/'
                     'semeval2016-task6-trainingdata.txt')
        file_path2 = ('data/semeval2016/'
                      'semeval2016-task6-trialdata.txt')
    elif split == 'test':
        file_path = ('data/semeval2016/'
                     'SemEval2016-Task6-subtaskA-testdata-gold.txt')

    # read data
    data = []
    with open(file_path, 'r', encoding='windows-1252') as f:
        for row in tqdm(f.readlines()[1:],
                        desc=f'loading SemEval2016 {split}ing data'):
            _, target, claim, stance = row.split('\t')
            data.append([target, claim, stance])

    # read train data for another file
    if split == 'train':
        with open(file_path2, 'r', encoding='windows-1252') as f:
            for row in f.readlines()[1:]:
                _, target, claim, stance = row.split('\t')
                data.append([target, claim, stance])

    # preprocessing
    data = preprocessing(data)

    # convert to dataframe
    data_df = convert_to_dataframe(data)

    return data_df  # target, claim, stance

def load_dataset_fnc(split='train'):
    # file path
    if split == 'train':
        target_path = 'data/fnc-1/train_stances.csv'
        claim_path = 'data/fnc-1/train_bodies.csv'
    elif split == 'test':
        target_path = 'data/fnc-1/competition_test_stances.csv'
        claim_path = 'data/fnc-1/competition_test_bodies.csv'

    # read data
    data = []
    target_df = pd.read_csv(target_path)
    claim_df = pd.read_csv(claim_path)

    for _, row in tqdm(target_df.iterrows(), 
                       desc=f'loading FNC-1 {split}ing data',
                       total=len(target_df)):
        claim = claim_df.loc[claim_df['Body ID'] == row['Body ID'], 
                             'articleBody'].tolist()[0]
        data.append([row['Headline'], claim, row['Stance']])

    # preprocessing
    data = preprocessing(data)

    # convert to dataframe
    data_df = convert_to_dataframe(data)

    return data_df  # target, claim, stance

def load_dataset_mnli(split='train'):
    # file path
    if split == 'train':
        file_path = ('data/multinli/'
                     'multinli_1.0_train.txt')
    elif split == 'test':
        file_path = ('data/multinli/'
                     'multinli_1.0_dev_matched.txt')

    # read data
    data = []
    with open(file_path, 'r') as f:
        for row in tqdm(f.readlines()[1:], 
                        desc=f'loading MultiNLI {split}ing data'):
            row = row.split('\t')
            data.append([row[5], row[6], row[0]])

    # preprocessing
    data = preprocessing(data)

    # convert to dataframe
    data_df = convert_to_dataframe(data)

    return data_df  # premise, hypothesis, label

def load_dataset(dataset=None):
    # load dataset by passed parameter
    if dataset == 'semeval2016_train':
        return load_dataset_semeval2016(split='train')
    elif dataset == 'semeval2016_test':
        return load_dataset_semeval2016(split='test')

    if dataset == 'fnc_train':
        return load_dataset_fnc(split='train')
    elif dataset == 'fnc_test':
        return load_dataset_fnc(split='test')

    if dataset == 'mnli_train':
        return load_dataset_mnli(split='train')
    elif dataset == 'mnli_test':
        return load_dataset_mnli(split='test')

    raise ValueError(f'dataset {dataset} does not support')

def load_lexicon_emolex(types='emotion'):
    # file path
    file_path = ('data/emolex/'
                 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

    # read data
    lexicons = []
    with open(file_path, 'r') as f:
        for row in tqdm(f.readlines()[1:], 
                        desc=f'loading EmoLex lexicon data'):
            word, emotion, value = row.split('\t')
            if types == 'emotion':
                if emotion not in ['negative', 'positive'] and int(value) == 1:
                    lexicons.append(word.strip())
            elif types == 'sentiment':
                if emotion in ['negative', 'positive'] and int(value) == 1:
                    lexicons.append(word.strip())

    lexicons = list(set(lexicons))

    return lexicons

def load_lexicon(lexicon=None):
    # load lexicon by passed parameter
    if lexicon == 'emolex_emotion':
        return load_lexicon_emolex(types='emotion')
    elif lexicon == 'emolex_sentiment':
        return load_lexicon_emolex(types='sentiment')

    raise ValueError(f'lexicon {lexicon} does not support')

class MultiTaskBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, datasets, batch_size, random_seed):
        self.datasets = datasets
        self.batch_size = batch_size
        self.random_seed = random_seed

        data_batch_index_list = []
        for dataset in datasets:
            data_batch_index_list.append(
                self.get_shuffled_batch_index(len(dataset), batch_size))

        self.data_batch_index_list = data_batch_index_list

    @staticmethod
    def get_shuffled_batch_index(dataset_len, batch_size):
        # get all index and shuffle them
        index_list = list(range(dataset_len))
        random.shuffle(index_list)

        # get batch index
        batches_index = []

        for i in range(0, dataset_len, batch_size):
            batch_index = []
            for j in range(i, min(i+batch_size, dataset_len)):
                batch_index.append(index_list[j])
            batches_index.append(batch_index)

        # shuffle batch index
        random.shuffle(batches_index)

        return batches_index

    @staticmethod
    def get_task_indices(data_batch_index_list, random_seed):
        all_indices = []

        for task_id, dataset in enumerate(data_batch_index_list):
            all_indices += [task_id]*len(dataset)

        # shuffle task indices
        random.seed(random_seed)  # set random seed
        random.shuffle(all_indices)

        return all_indices

    def __len__(self):
        return sum(len(data) for data in self.data_batch_index_list)

    def __iter__(self):
        all_iters = [iter(task_data)
                     for task_data in self.data_batch_index_list]
        all_task_indices = self.get_task_indices(self.data_batch_index_list,
                                                 self.random_seed)

        for task_id in all_task_indices:
            batch = next(all_iters[task_id])
            yield [(task_id, data_id) for data_id in batch]

class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(self, task_id,
                 target_encode, claim_encode,
                 claim_lexicon, label_encode):
        # 0 for stance detection and 1 for NLI
        self.task_id = task_id
        self.x1 = [torch.LongTensor(ids) for ids in target_encode]
        self.x2 = [torch.LongTensor(ids) for ids in claim_encode]
        self.lexicon = [torch.FloatTensor(ids) for ids in claim_lexicon]
        self.y = torch.LongTensor([label for label in label_encode])

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, index):
        return (self.task_id, self.x1[index], self.x2[index],
                self.lexicon[index], self.y[index])

    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        task_id = batch[0][0]
        x1 = [data[1] for data in batch]
        x2 = [data[2] for data in batch]
        lexicon = [data[3] for data in batch]
        y = torch.LongTensor([data[4] for data in batch])

        # pad sequence to fixed length with pad_token_id
        x1 = torch.nn.utils.rnn.pad_sequence(x1,
                                             batch_first=True,
                                             padding_value=pad_token_id)
        x2 = torch.nn.utils.rnn.pad_sequence(x2,
                                             batch_first=True,
                                             padding_value=pad_token_id)

        # pad lexicon to fixed length with value "0.0"
        lexicon = torch.nn.utils.rnn.pad_sequence(lexicon,
                                                  batch_first=True,
                                                  padding_value=0.0)

        return task_id, x1, x2, lexicon, y

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        task_id_to_dataset = {}

        for dataset in datasets:
            task_id = dataset.task_id
            task_id_to_dataset[task_id] = dataset

        self.task_id_to_dataset = task_id_to_dataset

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        task_id, data_id = index
        return self.task_id_to_dataset[task_id][data_id]