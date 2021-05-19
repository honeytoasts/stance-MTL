# built-in module
import argparse
import unicodedata
import re
import random
import os

# 3rd-party module
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pickle

# self-made module
from . import config
from . import tokenizer
from . import embedding

class MultiData:
    def __init__(self,
                 training: bool,
                 stance_target: str,
                 config: argparse.Namespace,
                 tokenizer: tokenizer.BaseTokenizer,
                 embedding: embedding.BaseEmbedding):

        # initialize dataframe
        data_df = pd.DataFrame(
            columns=['task_id', 'is_train', 'target_orig', 'claim_orig', 'label'])

        # load data
        stance_df = self.load_stance_data()
        nli_df = self.load_nli_data()
        data_df = pd.concat([data_df, stance_df, nli_df])

        # give ID for each raw data
        data_df['data_id'] = range(len(data_df))

        self.data = data_df

        # data preprocessing
        self.preprocessing()

        # if training then init tokenizer and embedding
        if training:
            # build vocabulary
            all_sentences = []
            all_sentences.extend(self.data['target'].tolist())
            all_sentences.extend(self.data['claim'].tolist())

            all_task_ids = []
            all_task_ids.extend(self.data['task_id'].tolist())

            tokenizer.build_vocabulary(all_sentences, all_task_ids)

            # get embeddings
            if 'fasttext' in config.embedding:
                embedding.load_fasttext_embedding(
                    id_to_token=tokenizer.id_to_token)
            else:
                embedding.load_file_embedding(
                    id_to_token=tokenizer.id_to_token)

        # content encode
        self.data['task_target_encode'], self.data['shared_target_encode'] = (
            tokenizer.encode(sentences=self.data['target'].tolist(),
                             task_ids=self.data['task_id']))
        self.data['task_claim_encode'], self.data['shared_claim_encode'] = (
            tokenizer.encode(sentences=self.data['claim'].tolist(),
                             task_ids=self.data['task_id']))

        # label encode
        label_to_id = {'favor': 0, 'against': 1, 'none': 2,
                       'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.data['label_encode'] = self.data['label'].apply(
            lambda label: label_to_id[label])

        # get attention mask
        task_attn_mask, shared_attn_mask = (
            self.get_attention_mask(pad_token_id=tokenizer.pad_token_id))
        self.data['task_attn_mask'] = task_attn_mask
        self.data['shared_attn_mask'] = shared_attn_mask

        # separate dataset
        self.stance_train_df = self.data[(self.data['task_id'] == 0) &
                                         (self.data['is_train'] == True)]
        self.stance_test_df = self.data[(self.data['task_id'] == 0) &
                                        (self.data['is_train'] == False)]
        self.nli_train_df = self.data[(self.data['task_id'] == 1) &
                                      (self.data['is_train'] == True)]
        self.nli_test_df = self.data[(self.data['task_id'] == 1) &
                                     (self.data['is_train'] == False)]
        
        # get specific dataset for stance
        if stance_target != 'all':
            self.stance_train_df = self.stance_train_df[
                self.stance_train_df['target_orig'] == stance_target]
            self.stance_test_df = self.stance_test_df[
                self.stance_test_df['target_orig'] == stance_target]

        # reset index
        self.stance_train_df = self.stance_train_df.reset_index(drop=True)
        self.stance_test_df = self.stance_test_df.reset_index(drop=True)
        self.nli_train_df = self.nli_train_df.reset_index(drop=True)
        self.nli_test_df = self.nli_test_df.reset_index(drop=True)

        # delete unused data
        del self.data
        
    def load_stance_data(self):
        # load SemEval data
        file_paths = [
            'data/semeval2016/semeval2016-task6-trialdata.txt',
            'data/semeval2016/semeval2016-task6-trainingdata.txt',
            'data/semeval2016/SemEval2016-Task6-subtaskA-testdata-gold.txt']

        stance_df = pd.DataFrame()
        for file_path in file_paths:
            temp_df = pd.read_csv(file_path, encoding='windows-1252', delimiter='\t')
            stance_df = pd.concat([stance_df, temp_df])
        stance_df.columns = ['ID', 'target_orig', 'claim_orig', 'label']

        # add task_id and is_train column
        stance_df['is_train'] = stance_df['ID'].apply(
            lambda idx: True if int(idx) < 10000 else False)
        stance_df['task_id'] = [0] * len(stance_df)
        stance_df = stance_df[['task_id', 'is_train',
                               'target_orig', 'claim_orig', 'label']]

        return stance_df

    def load_nli_data(self):
        # load MNLI data
        file_paths = [
            'data/multinli/multinli_1.0_train.jsonl',
            'data/multinli/multinli_1.0_dev_matched.jsonl']

        nli_df = pd.DataFrame()
        for idx, file_path in enumerate(file_paths):
            temp_df = pd.read_json(file_path, lines=True)
            temp_df = temp_df[['sentence1', 'sentence2', 'gold_label']]
            temp_df['is_train'] = [True if not idx else False] * len(temp_df)
            nli_df = pd.concat([nli_df, temp_df])
        nli_df.columns = ['target_orig', 'claim_orig', 'label', 'is_train']

        # add task_id column
        nli_df['task_id'] = [1] * len(nli_df)
        nli_df = nli_df[['task_id', 'is_train',
                         'target_orig', 'claim_orig', 'label']]

        # remove the data which label is '-'
        nli_df = nli_df[nli_df['label'] != '-']

        return nli_df

    def preprocessing(self):
        # encoding normalize
        normalize_func = (
            lambda text: unicodedata.normalize('NFKC', str(text)))

        self.data['target'] = self.data['target_orig'].apply(normalize_func)
        self.data['claim'] = self.data['claim_orig'].apply(normalize_func)
        self.data['label'] = self.data['label'].apply(normalize_func)

        # tweet preprocessing
        self.data['claim'] = self.data['claim'].apply(
            self.tweet_preprocessing)

        # change to lower case
        lower_func = lambda text: text.lower().strip()

        self.data['target'] = self.data['target'].apply(lower_func)
        self.data['claim'] = self.data['claim'].apply(lower_func)
        self.data['label'] = self.data['label'].apply(lower_func)

    def tweet_preprocessing(self, text):
        # reference: https://github.com/zhouyiwei/tsd/blob/master/utils.py

        text = text.replace('#SemST', '').strip()

        text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<URL>", text)
        text = re.sub(r"@\w+", "<USER>", text)
        text = re.sub(r"[8:=;]['`\-]?[)d]+|[)d]+['`\-]?[8:=;]", "<SMILE>", text)
        text = re.sub(r"[8:=;]['`\-]?p+", "<LOLFACE>", text)
        text = re.sub(r"[8:=;]['`\-]?\(+|\)+['`\-]?[8:=;]", "<SADFACE>", text)
        text = re.sub(r"[8:=;]['`\-]?[\/|l*]", "<NEUTRALFACE>", text)
        text = re.sub(r"<3","<HEART>", text)
        text = re.sub(r"/"," / ", text)
        text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
        p = re.compile(r"#\S+")
        text = p.sub(lambda s: "<HASHTAG> "+s.group()[1:]+" <ALLCAPS>"
                     if s.group()[1:].isupper()
                     else " ".join(["<HASHTAG>"]+re.split(r"([A-Z][^A-Z]*)",
                                   s.group()[1:])),text)
        text = re.sub(r"([!?.]){2,}", r"\1 <REPEAT>", text)
        text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <ELONG>", text)

        return text

    def get_attention_mask(self, pad_token_id):
        task_claim_ids = torch.tensor(self.data['task_claim_encode'].tolist())
        shared_claim_ids = torch.tensor(self.data['shared_claim_encode'].tolist())

        # 1 for attention, 0 for mask
        task_attn_mask = (task_claim_ids != pad_token_id)
        shared_attn_mask = (shared_claim_ids != pad_token_id)

        return task_attn_mask.tolist(), shared_attn_mask

class MultiTaskBatchSampler(torch.utils.data.BatchSampler):
    # reference: https://github.com/namisan/mt-dnn/blob/master/mt_dnn/batcher.py
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

class MultiTaskDataset(torch.utils.data.Dataset):
    # reference: https://github.com/namisan/mt-dnn/blob/master/mt_dnn/batcher.py
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

class SingleTaskDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_id: pd.Series,
                 task_id: int,
                 target_name: pd.Series,
                 task_target: pd.Series,
                 shared_target: pd.Series,
                 task_claim: pd.Series,
                 shared_claim: pd.Series,
                 task_attn_mask: pd.Series,
                 shared_attn_mask: pd.Series,
                 labels: pd.Series):

        self.data_id = [idx for idx in data_id]
        self.task_id = task_id
        self.target_name = target_name.reset_index(drop=True)
        self.task_target = [ids for ids in task_target]
        self.shared_target = [ids for ids in shared_target]
        self.task_claim = [ids for ids in task_claim]
        self.shared_claim = [ids for ids in shared_claim]
        self.task_attn_mask = [mask for mask in task_attn_mask]
        self.shared_attn_mask = [mask for mask in shared_attn_mask]
        self.label = [label for label in labels]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get encode length
        task_target_len = len(self.task_target[index])
        task_total_len = task_target_len+len(self.task_claim[index])

        shared_target_len = len(self.shared_target[index])
        shared_total_len = shared_target_len+len(self.shared_claim[index])

        # load dependency relation
        data_id = self.data_id[index]

        # prevent cannot load deprel file
        if os.path.exists(f'data/dep_relation/deprel_{data_id}.pickle'):
            with open(f'data/dep_relation/deprel_{data_id}.pickle', 'rb') as f:
                dep_rel = pickle.load(f)
        else:
            print(f'deprel_{data_id}.pickle is not found')
            dep_rel = {}

        # get adjacency matrix
        task_adj_matrix, shared_adj_matrix = (
            self.get_adj_matrix(task_target_len,
                                task_total_len,
                                shared_target_len,
                                shared_total_len,
                                dep_rel))

        return (self.data_id[index], self.task_id,
                self.target_name[index],
                self.task_target[index], self.shared_target[index],
                self.task_claim[index], self.shared_claim[index],
                self.task_attn_mask[index], self.shared_attn_mask[index],
                task_adj_matrix, shared_adj_matrix,
                self.label[index])

    def get_adj_matrix(self,
                       task_target_len: int,
                       task_total_len: int,
                       shared_target_len: int,
                       shared_total_len: int,
                       dep_rel: dict):

        # initialize adjacency matrix
        task_matrix = np.array([[0 for _ in range(task_total_len)] 
                                for _ in range(task_total_len)])
        shared_matrix = np.array([[0 for _ in range(shared_total_len)]
                                  for _ in range(shared_total_len)])

        # self-connected: let diagonal elements equal to 1
        np.fill_diagonal(task_matrix, 1)
        np.fill_diagonal(shared_matrix, 1)

        # iterate relation if dep_rel is not emoty
        if dep_rel != {}:
            # iterate target's dependency relation
            for word in dep_rel['target_dep']:
                if word['id'] >= 1 and word['head'] >= 1:
                    # get index
                    index_a = word['id']-1
                    index_b = word['head']-1

                    # add edge to "task" adjacency matrix
                    if index_a < task_target_len and (
                       index_b < task_target_len):
                        task_matrix[index_a][index_b] = 1
                        task_matrix[index_b][index_a] = 1

                    # add edge to "shared" adjacency matrix
                    if index_a < shared_target_len and (
                       index_b < shared_target_len):
                        shared_matrix[index_a][index_b] = 1
                        shared_matrix[index_b][index_a] = 1

            # iterate claim's dependency relation
            for word in dep_rel['claim_dep']:
                if word['id'] >= 1 and word['head'] >= 1:
                    # get "task" index
                    task_index_a = task_target_len+word['id']-1
                    task_index_b = task_target_len+word['head']-1

                    # get "shared" index
                    shared_index_a = shared_target_len+word['id']-1
                    shared_index_b = shared_target_len+word['head']-1

                    # add edge to "task" adjacency matrix
                    if task_index_a < task_total_len and (
                       task_index_b < task_total_len):
                        task_matrix[task_index_a][task_index_b] = 1
                        task_matrix[task_index_b][task_index_a] = 1

                    # add edge to "shared" adjacency matrix
                    if shared_index_a < shared_total_len and (
                       shared_index_b < shared_total_len):
                        shared_matrix[shared_index_a][shared_index_b] = 1
                        shared_matrix[shared_index_b][shared_index_a] = 1

        return task_matrix.tolist(), shared_matrix.tolist()

# reference: https://discuss.pytorch.org/t/supplying-arguments-to-collate-fn/25754/3
class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        task_id = batch[0][1]
        target_name = [data[2] for data in batch]
        task_target = [torch.LongTensor(data[3]) for data in batch]
        shared_target = [torch.LongTensor(data[4]) for data in batch]
        task_claim = [torch.LongTensor(data[5]) for data in batch]
        shared_claim = [torch.LongTensor(data[6]) for data in batch]
        task_attn_mask = torch.LongTensor([data[7] for data in batch])
        shared_attn_mask = torch.LongTensor([data[8] for data in batch])
        task_adj_matrix = torch.LongTensor([data[9] for data in batch])
        shared_adj_matrix = torch.LongTensor([data[10] for data in batch])
        label = torch.LongTensor([data[11] for data in batch])

        # pad target sequence
        task_target = (
            torch.nn.utils.rnn.pad_sequence(
                task_target,
                batch_first=True,
                padding_value=self.pad_token_id))
        shared_target = (
            torch.nn.utils.rnn.pad_sequence(
                shared_target,
                batch_first=True,
                padding_value=self.pad_token_id))

        # pad target sequence
        task_claim = (
            torch.nn.utils.rnn.pad_sequence(
                task_claim,
                batch_first=True,
                padding_value=self.pad_token_id))
        shared_claim = (
            torch.nn.utils.rnn.pad_sequence(
                shared_claim,
                batch_first=True,
                padding_value=self.pad_token_id))

        return (task_id,
                target_name,
                task_target, shared_target,
                task_claim, shared_claim,
                task_attn_mask, shared_attn_mask,
                task_adj_matrix, shared_adj_matrix,
                label)