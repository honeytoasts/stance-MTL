# built-in module
import re
import pickle

# 3rd-party module
import torch

# self-made module
import tokenizers

class Embedding:
    def __init__(self, tokenizer: tokenizers.Tokenizer, embedding_dim, random_seed=7):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.word_dict = {}
        self.vector = torch.Tensor()
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)

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
        torch.manual_seed(self.random_seed)

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

        with open(embedding_path) as f:
            firstrow = f.readline()
            # if first row not the header
            if len(firstrow.strip().split()) >= self.embedding_dim:
                # seek to 0
                f.seek(0)

            for row in f:
                # get token and embedding
                row = row.strip().split()
                token = row[0]

                # add embedding if token is provided
                if token in tokens and token not in self.word_dict:
                    self.word_dict[token] = len(self.word_dict)
                    vectors.append([float(v) for v in row[1:]])

        vectors = torch.Tensor(vectors)
        self.vector = torch.cat([self.vector, vectors], dim=0)