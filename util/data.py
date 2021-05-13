# built-in module
import argparse
import unicodedata
import re
import gc

# 3rd-party module
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

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
        stance_df = self.load_stance_data(stance_target)
        nli_df = self.load_nli_data()
        data_df = pd.concat([data_df, stance_df, nli_df])

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
        self.data['target_encode'] = tokenizer.encode(
            sentences=self.data['target'].tolist(),
            task_ids=self.data['task_id'])
        self.data['claim_encode'] = tokenizer.encode(
            sentences=self.data['claim'].tolist(),
            task_ids=self.data['task_id'])

        # label encode
        label_to_id = {'favor': 0, 'against': 1, 'none': 2,
                 'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.data['label_encode'] = self.data['label'].apply(
            lambda label: label_to_id[label])

        # separate two dataset
        self.stance_train_df = self.data[(self.data['task_id'] == 0) &
                                         (self.data['is_train'] == True)]
        self.stance_test_df = self.data[(self.data['task_id'] == 0) &
                                        (self.data['is_train'] == False)]
        self.nli_train_df = self.data[(self.data['task_id'] == 1) &
                                      (self.data['is_train'] == True)]
        self.nli_test_df = self.data[(self.data['task_id'] == 1) &
                                     (self.data['is_train'] == False)]

        # reset index
        self.stance_train_df = self.stance_train_df.reset_index(drop=True)
        self.stance_test_df = self.stance_test_df.reset_index(drop=True)
        self.nli_train_df = self.nli_train_df.reset_index(drop=True)
        self.nli_test_df = self.nli_test_df.reset_index(drop=True)
        
    def load_stance_data(self, target):
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

        # get specific target data
        if target != 'all':
            stance_df = stance_df[stance_df['target_orig'] == target]

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