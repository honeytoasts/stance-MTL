# built-in module
import re
import os
import pickle

# 3rd-party module
from tqdm import tqdm

class Tokenizer:
    def __init__(self):
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

        self.token_to_id = {}
        self.id_to_token = {}
        self.all_tokens = []

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

    def get_all_tokens(self, sentences):
        sentences = self.tokenize(sentences)
        tokens = []

        for sentence in tqdm(sentences, desc='get all tokens'):
            for token in sentence:
                tokens.append(token)

        tokens = list(set(tokens))
        self.all_tokens = tokens

    def tokenize(self, sentences):
        # space tokenize
        return [sentence.split() for sentence in sentences]

    def detokenize(self, sentences):
        return [' '.join(sentence) for sentence in sentences]

    def build_dict(self, sentences, word_dict):
        sentences = self.tokenize(sentences)

        for token in self.all_tokens:
            if token not in self.token_to_id and token in word_dict:
                self.token_to_id[token] = word_dict[token]
                self.id_to_token[word_dict[token]] = token
            elif token not in self.token_to_id and token not in word_dict:
                self.token_to_id[token] = self.unk_token_id

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

        for sentence in sentences:
            sentence.insert(0, self.bos_token_id)
            sentence.append(self.eos_token_id)

        return sentences

    def decode(self, sentences):
        sentences = self.detokenize(self.convert_ids_to_tokens(sentences))
        # sentences = [re.sub(r'[bos]|[sep]|[eos]|[pad]', '', sentence)
        #              for sentence in sentences]

        return sentences

    def load_from_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            self.token_to_id = pickle.load(f)
        self.id_to_token = {v:i for i, v in self.token_to_id.items()}

        return self

    def save_to_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self.token_to_id, f)

        return self

    def encode_to_lexicon(self, sentences, lexicons):
        sentences = self.tokenize(sentences)
        lexicons = {lexicon: 1 for lexicon in lexicons}
        lexicon_vectors = []

        for sentence in sentences:
            lexicon_vector = [lexicons.get(token, 0) for token in sentence]

            # bos and eos token have no lexicon
            lexicon_vector.insert(0, 0)
            lexicon_vector.append(0)

            lexicon_vectors.append(lexicon_vector)

        return lexicon_vectors