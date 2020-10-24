# built-in module
import os
import pickle

# 3rd-party module
import torch
from tqdm import tqdm

class BaseEmbedding:
    def __init__(self, embedding_dim=200):
        self.embedding_dim = embedding_dim
        self.word_dict = {}
        self.vector = torch.Tensor()

        if '[pad]' not in self.word_dict:
            self.add_embedding('[pad]', torch.zeros(self.embedding_dim))
        if '[bos]' not in self.word_dict:
            self.add_embedding('[bos]')
        if '[sep]' not in self.word_dict:
            self.add_embedding('[sep]')
        if '[eos]' not in self.word_dict:
            self.add_embedding('[eos]')
        if '[unk]' not in self.word_dict:
            self.add_embedding('[unk]')

    def get_num_embeddings(self):
        return self.vector.shape[0]

    def add_embedding(self, token, vector=None):
        if vector is not None:
            vector = vector.unsqueeze(0)
        else:
            vector = torch.empty(1, self.embedding_dim)
            torch.nn.init.normal_(vector, mean=0, std=1)

        self.word_dict[token] = len(self.word_dict)
        self.vector = torch.cat([self.vector, vector], dim=0)

    def load_embedding(self, embedding_path, tokens):
        tokens = set(tokens)
        vectors = []

        # get number of rows in file
        with open(embedding_path) as f:
            file_len = len(f.readlines())

        with open(embedding_path) as f:
            firstrow = f.readline()

            # if first row not the header, seek to 0
            if len(firstrow.strip().split()) >= self.embedding_dim:
                f.seek(0)

            for row in tqdm(f, desc='load embedding', total=file_len):
                # get token and embedding
                row = row.strip().split()
                token = row[0]

                # add embedding if token is provided
                if token in tokens and token not in self.word_dict:
                    self.word_dict[token] = len(self.word_dict)
                    vectors.append([float(v) for v in row[1:]])

        vectors = torch.Tensor(vectors)
        self.vector = torch.cat([self.vector, vectors], dim=0)

    def load(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            embedding = pickle.load(f)
            self.embedding_dim = embedding.embedding_dim
            self.word_dict = embedding.word_dict
            self.vector = embedding.vector

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)