# built-in modules
import os
import json

class BaseConfig:
    def __init__(self, **kwargs):
        # experiment no
        self.experiment_no = kwargs.pop('experiment_no', 1)

        # preprocess
        self.tokenizer = kwargs.pop('tokenizer', 'BaseTokenizer')
        self.filter = kwargs.pop('filter', 'none')
        self.min_count = kwargs.pop('min_count', 1)
        self.max_seq_len = kwargs.pop('max_seq_len', 20)

        # dataset and lexicon
        self.stance_dataset = kwargs.pop('stance_dataset', 'semeval2016')
        self.embedding_file = kwargs.pop('embedding_file', 'glove')
        self.lexicon_file = kwargs.pop('lexicon_file', 'emolex_emotion')
        self.stance_output_dim = 3 if 'semeval' in self.stance_dataset else 4
        self.nli_output_dim = 3

        # hyperparameter
        self.embedding_dim = kwargs.pop('embedding_dim', 1)
        self.task_hidden_dim = kwargs.pop('task_hidden_dim', 1)
        self.shared_hidden_dim = kwargs.pop('shared_hidden_dim', 1) 

        self.num_rnn_layers = kwargs.pop('num_rnn_layers', 1)
        self.num_linear_layers = kwargs.pop('num_linear_layers', 1)
        self.attention = kwargs.pop('attention', 'dot')
        self.dropout = kwargs.pop('dropout', 0)

        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.clip_grad_value = kwargs.pop('clip_grad_value', 0)
        self.weight_decay = kwargs.pop('weight_decay', 0)
        self.lr_decay_step = kwargs.pop('lr_decay_step', 10)
        self.lr_decay = kwargs.pop('lr_decay', 1)

        self.nli_loss_weight = kwargs.pop('nli_loss_weight', 1.0)
        self.lexicon_loss_weight = kwargs.pop('lexicon_loss_weight', 0)

        # others
        self.random_seed = kwargs.pop('random_seed', 77)
        self.kfold = kwargs.pop('kfold', 5)
        self.train_test_split = kwargs.pop('train_test_split', 0.15)
        self.epoch = kwargs.pop('epoch', 50)
        self.batch_size = kwargs.pop('batch_size', 32)

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            # save config file
            with open(file_path, 'w') as f:
                hyperparameters = {
                    'experiment_no': self.experiment_no,
                    'tokenizer': self.tokenizer,
                    'filter': self.filter,
                    'min_count': self.min_count,
                    'max_seq_len': self.max_seq_len,
                    'stance_dataset': self.stance_dataset,
                    'embedding_file': self.embedding_file,
                    'lexicon_file': self.lexicon_file,
                    'stance_output_dim': self.stance_output_dim,
                    'nli_output_dim': self.nli_output_dim,
                    'embedding_dim': self.embedding_dim,
                    'task_hidden_dim': self.task_hidden_dim,
                    'shared_hidden_dim': self.shared_hidden_dim,
                    'num_rnn_layers': self.num_rnn_layers,
                    'num_linear_layers': self.num_linear_layers,
                    'attention': self.attention,
                    'dropout': self.dropout,
                    'learning_rate': self.learning_rate,
                    'clip_grad_value': self.clip_grad_value,
                    'weight_decay': self.weight_decay,
                    'lr_decay_step': self.lr_decay_step,
                    'lr_decay': self.lr_decay,
                    'nli_loss_weight': self.nli_loss_weight,
                    'lexicon_loss_weight': self.lexicon_loss_weight,
                    'random_seed': self.random_seed,
                    'kfold': self.kfold,
                    'train_test_split': self.train_test_split,
                    'epoch': self.epoch,
                    'batch_size': self.batch_size
                }

                json.dump(hyperparameters, f)

    @classmethod
    def load(cls, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        # load config file
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))