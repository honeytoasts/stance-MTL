# 3rd-party module
import os
import pandas as pd
import unicodedata
import re
import nltk
import stanza
import gc
import torch
from tqdm import tqdm
import pickle

# define class Data
class Data:
    def __init__(self):

        # initialize dataframe
        data_df = pd.DataFrame(
            columns=['task_id','target', 'claim'])

        # load data
        stance_df = self.load_stance_data()
        nli_df = self.load_nli_data()
        data_df = pd.concat([data_df, stance_df, nli_df])

        # give ID for each raw data
        data_df['ID'] = range(len(data_df))

        self.data = data_df

        # data preprocessing
        self.preprocessing()

    def load_stance_data(self):
        # load SemEval data
        file_paths = [
            'data/semeval2016/semeval2016-task6-trialdata.txt',
            'data/semeval2016/semeval2016-task6-trainingdata.txt',
            'data/semeval2016/SemEval2016-Task6-subtaskA-testdata-gold.txt']

        stance_df = pd.DataFrame()
        for file_path in file_paths:
            temp_df = pd.read_csv(file_path, encoding='windows-1252', delimiter='\t')
            temp_df = temp_df[['Target', 'Tweet']]
            stance_df = pd.concat([stance_df, temp_df])
        stance_df.columns = ['target', 'claim']

        # add task_id and is_train column
        stance_df['task_id'] = [0] * len(stance_df)

        # adjust column order
        stance_df = stance_df[['task_id', 'target', 'claim']]

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
            nli_df = pd.concat([nli_df, temp_df])
        nli_df.columns = ['target', 'claim', 'label']

        # add task_id column
        nli_df['task_id'] = [1] * len(nli_df)

        # remove the data which label is '-'
        nli_df = nli_df[nli_df['label'] != '-']

        # adjust column order
        nli_df = nli_df[['task_id', 'target', 'claim']]

        return nli_df

    def preprocessing(self):
        # encoding normalize
        normalize_func = (
            lambda text: unicodedata.normalize('NFKC', str(text)))

        self.data['target'] = self.data['target'].apply(normalize_func)
        self.data['claim'] = self.data['claim'].apply(normalize_func)

        # tweet preprocessing
        self.data['claim'] = self.data['claim'].apply(
            self.tweet_preprocessing)

        # change to lower case
        lower_func = lambda text: text.lower().strip()

        self.data['target'] = self.data['target'].apply(lower_func)
        self.data['claim'] = self.data['claim'].apply(lower_func)

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

# define tokenize function
def tokenize(sentences, task_ids):
    # nltk TweetTokenizer for stance
    tweet_tokenizer = nltk.tokenize.TweetTokenizer()

    # nltk WordPunctTokenizer for NLI
    punct_tokenizer = nltk.tokenize.WordPunctTokenizer()

    all_sentence = []
    for sentence, task_id in zip(sentences, task_ids):
        if task_id == 0:  # stance
            tokenize_sent = tweet_tokenizer.tokenize(sentence)
        elif task_id == 1:  # NLI
            tokenize_sent = punct_tokenizer.tokenize(sentence)

        all_sentence.append(tokenize_sent)

    return all_sentence

def main():
    # create directory if not exists
    if not os.path.exists('data/dep_relation'):
        os.makedirs('data/dep_relation')

    # get data
    data_df = Data().data

    # tokenize sentence
    tokenized_target = tokenize(data_df['target'], data_df['task_id'])
    tokenized_claim = tokenize(data_df['claim'], data_df['task_id'])

    # initialize dependency parser
    dep_config = {
        'processors': 'tokenize,pos,lemma,depparse',
        'tokenize_pretokenized': True,
        'lang': 'en',
        'dir': 'data/stanza',
        'use_gpu': True,
        'logging_level': 'ERROR'}
    dep_parser = stanza.Pipeline(**dep_config)

    # get target dependency relation
    print('get dependncy relation of target sentences--')
    dep_target = dep_parser(tokenized_target)

    # release GPU memory
    del dep_parser
    gc.collect()
    torch.cuda.empty_cache()

    # initialize another dependency parser
    dep_config = {
        'processors': 'tokenize,pos,lemma,depparse',
        'tokenize_pretokenized': True,
        'lang': 'en',
        'dir': 'data/stanza',
        'use_gpu': True,
        'logging_level': 'ERROR'}
    dep_parser = stanza.Pipeline(**dep_config)

    # get claim dependency relation
    print('get dependncy relation of claim sentences--')
    dep_claim = dep_parser(tokenized_claim)

    # release GPU memory
    del dep_parser
    gc.collect()
    torch.cuda.empty_cache()

    # insert to dataframe
    data_df['target_dep'] = dep_target.sentences
    data_df['claim_dep'] = dep_claim.sentences

    # save dataframe for each raw data
    for _, row in tqdm(data_df.iterrows(),
                       total=len(data_df),
                       desc='save dependency relation'):

        target_dep = [{'id': word.id, 'head': word.head, 'deprel': word.deprel}
                      for word in row['target_dep'].words]
        claim_dep = [{'id': word.id, 'head': word.head, 'deprel': word.deprel}
                      for word in row['claim_dep'].words]
        dep_rel = {'target_dep': target_dep, 'claim_dep': claim_dep}

        with open(f"data/dep_relation/deprel_{row['ID']}.pickle", 'wb') as f:
            pickle.dump(dep_rel, f)

    return

if __name__ == '__main__':
    main()
