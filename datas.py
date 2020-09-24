# built-in module
import re
import unicodedata

# 3rd-party module
import pandas as pd
import numpy as np

def load_dataset_semeval2016(split='train'):
    # define file path
    if split == 'train':
        file_path = 'data/semeval2016_train.txt'
    elif split == 'test':
        file_path = 'data/semeval2016_test.txt'

    # read data
    data = []
    with open(file_path, 'r', encoding='windows-1252') as f:
        for row in f.readlines()[1:]:
            _, target, claim, stance = row.split('\t')
            data.append([target, claim, stance])

    # preprocessing
    # encoding normalize
    data = [[unicodedata.normalize('NFKC', str(column))
             for column in row] for row in data]

    # remove specific pattern
    pattern = r"@[a-zA-Z]+|[0-9]+\S[0-9]*|[0-9]+|[^a-zA-Z ']|RT"
    data = [[re.sub(pattern, '', column)
             for column in row] for row in data]

    # change to lowercase
    data = [[column.lower() for column in row] for row in data]

    return data

def load_dataset_fnc(split='train'):
    return

def load_dataset_mnli(split='train'):
    return

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