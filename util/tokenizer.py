# built-in module
import os
import pickle
from nltk.corpus import stopwords
from nltk import tokenize

# 3rd-party module
from tqdm import tqdm
import pandas as pd

class BaseTokenizer:
    def __init__(self, config):
        # padding token
        self.pad_token = '[pad]'
        self.pad_token_id = 0

        # begin of sentence token
        self.bos_token = '[bos]'
        self.bos_token_id = 1

        # separate token
        self.sep_token = '[sep]'
        self.sep_token_id = 2

        # end of sentence token
        self.eos_token = '[eos]'
        self.eos_token_id = 3

        # unknown token
        self.unk_token = '[unk]'
        self.unk_token_id = 4

        # others
        self.token_to_id = {}
        self.id_to_token = {}
        self.all_tokens = []
        self.lexicon_dict = {}
        self.config = config

        self.token_to_id[self.pad_token] = self.pad_token_id
        self.token_to_id[self.bos_token] = self.bos_token_id
        self.token_to_id[self.sep_token] = self.sep_token_id
        self.token_to_id[self.eos_token] = self.eos_token_id
        self.token_to_id[self.unk_token] = self.unk_token_id

        self.id_to_token[self.pad_token_id] = self.pad_token
        self.id_to_token[self.bos_token_id] = self.bos_token
        self.id_to_token[self.sep_token_id] = self.sep_token
        self.id_to_token[self.eos_token_id] = self.eos_token
        self.id_to_token[self.unk_token_id] = self.unk_token

        # stopwords 
        # ref: https://github.com/sheffieldnlp/stance-conditional/blob/master/stancedetection/preprocess.py
        if config.filter == 'all':
            stopword = stopwords.words('english')
            stopword.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",",
                             "-", ".", "/", ":", ";", "<", ">", "@", "[", "]", "^", 
                             "_", "`", "{", "|", "}", "~", "=", "+", "!",  "?"])
            stopword.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])

            self.stopwords = stopword
        elif config.filter == 'punctonly':
            stopword = ["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",",
                        "-", ".", "/", ":", ";", "<", ">", "@", "[", "]", "^", 
                        "_", "`", "{", "|", "}", "~", "=", "+", "!",  "?"]
            stopword.extend(["rt", "#semst", "..."])

            self.stopwords = stopword
        elif config.filter == 'none':
            self.stopwords = ["rt", "#semst", "..."]

    def tokenize(self, sentences):
        # space tokenizer
        sentences = [sentence.split() for sentence in sentences]

        # filter stopwords
        sentences = self.filter_stopwords(sentences)

        return sentences

    def detokenize(self, sentences):
        return [' '.join(sentence) for sentence in sentences]

    def filter_stopwords(self, sentences):
        result = []

        for sentence in sentences:
            result.append([token for token in sentence
                           if token not in self.stopwords])

        return result

    def get_all_tokens(self, sentences):
        # tokenize
        sentences = self.tokenize(sentences)

        # calculate word count
        words_count = {}
        for sentence in sentences:
            for token in sentence:
                words_count[token] = words_count.setdefault(token, 0) + 1

        # filter num of tokens < min_count
        tokens = [token for token, count in words_count.items()
                  if count >= self.config.min_count]
        self.all_tokens = tokens

        return self.all_tokens

    def build_dict(self, word_dict):
        self.token_to_id = word_dict
        self.id_to_token = {idx: token for token, idx in word_dict.items()}
        self.all_tokens = [word for word in word_dict]

    def build_lexicon_dict(self, lexicons):
        # convert lexicons token to id
        lexicon_dict = {}
        token_series = pd.Series(self.token_to_id.keys())

        for lexicon in tqdm(lexicons, desc='build lexicon dictionary'):
            # check whether lexicon in all_tokens
            lexicon_in_tokens = token_series.str.startswith(lexicon)
            lexicon_ids = lexicon_in_tokens[lexicon_in_tokens == True].index.tolist()

            for idx in lexicon_ids:
                lexicon_dict[idx] = 1

        self.lexicon_dict = lexicon_dict

    def convert_tokens_to_ids(self, sentences):
        result = []

        for sentence in sentences:
            ids = []
            for token in sentence:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    ids.append(self.unk_token_id)
            result.append(ids)

        return result

    def convert_ids_to_tokens(self, sentences):
        result = []

        for sentence in sentences:
            tokens = []
            for idx in sentence:
                if idx in self.id_to_token:
                    tokens.append(self.id_to_token[idx])
                else:
                    raise ValueError(f'idx {idx} not in the dictionary')
            result.append(tokens)

        return result

    def encode(self, sentences):
        sentences = self.convert_tokens_to_ids(self.tokenize(sentences))

        for i in range(len(sentences)):
            # add bos token id at the front
            sentence = [self.bos_token_id] + sentences[i]

            # cut off sentence
            sentence = sentence[:self.config.max_seq_len]

            # padding if sentence length < max sequence length
            pad_count = self.config.max_seq_len - len(sentence)
            sentence.extend([self.pad_token_id] * pad_count)

            # replace last token with eos token id
            sentence[-1] = self.eos_token_id

            sentences[i] = sentence

        return sentences

    def decode(self, sentences):
        sentences = self.detokenize(self.convert_ids_to_tokens(sentences))

        return sentences

    def encode_to_lexicon(self, sentences):
        # encode to lexicon vector
        lexicon_vectors = []

        for sentence in sentences:
            lexicon_vector = [self.lexicon_dict.get(token, 0)
                              for token in sentence]
            lexicon_vectors.append(lexicon_vector)      

        return lexicon_vectors

    def load(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            tokenizer = pickle.load(f)
            self.token_to_id = tokenizer.token_to_id
            self.id_to_token = tokenizer.id_to_token
            self.all_tokens = tokenizer.all_tokens

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)

class WordPunctTokenizer(BaseTokenizer):
    def __init__(self, config):
        super(WordPunctTokenizer, self).__init__(config)

    def tokenize(self, sentences):
        # nltk WordPunctTokenizer
        tokenizer = tokenize.WordPunctTokenizer()
        sentences = [tokenizer.tokenize(sentence) for sentence in sentences]

        # filter stopwords
        sentences = self.filter_stopwords(sentences)

        return sentences