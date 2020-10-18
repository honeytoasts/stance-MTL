# built-in modules
import os
import pickle

class Config:
    def __init__(self, **kwargs):
        self.stance_dataset = kwargs.pop('stance_dataset', 'semeval2016')
        self.embedding_file = kwargs.pop('embedding_file', 'glove')
        self.lexicon_file = kwargs.pop('lexicon_file', 'emolex_emotion')

        self.random_seed = kwargs.pop('random_seed', 7)
        self.epoch = kwargs.pop('epoch', 1)
        self.batch_size = kwargs.pop('batch_size', 1)
        self.learning_rate = kwargs.pop('learning_rate', 1e-3)
        self.kfold = kwargs.pop('kfold', 5)

        self.dropout = kwargs.pop('dropout', 0)
        self.embedding_dim = kwargs.pop('embedding_dim', 1)
        self.hidden_dim = kwargs.pop('hidden_dim', 1)
        self.stance_output_dim = kwargs.pop('stance_output_dim', 1)
        self.nli_output_dim = kwargs.pop('nli_output_dim', 1)
        self.num_rnn_layers = kwargs.pop('num_rnn_layers', 1)
        self.num_linear_layers = kwargs.pop('num_linear_layers', 1)
        self.attention = kwargs.pop('attention', 'dot')
        self.clip_grad_value = kwargs.pop('clip_grad_value', 1)
        self.nli_loss_weight = kwargs.pop('nli_loss_weight', 1.0)
        self.lexicon_loss_weight = kwargs.pop('lexicon_loss_weight', 0.025)

    def load_from_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        with open(file_path, 'rb') as f:
            hyperparameters = pickle.load(f)

            self.stance_dataset = \
                hyperparameters.pop('stance_dataset', self.stance_dataset)
            self.embedding_file = \
                hyperparameters.pop('embedding_file', self.embedding_file)
            self.lexicon_file = \
                hyperparameters.pop('lexicon_file', self.lexicon_file)

            self.random_seed = \
                hyperparameters.pop('random_seed', self.random_seed)
            self.epoch = \
                hyperparameters.pop('epoch', self.epoch)
            self.batch_size = \
                hyperparameters.pop('batch_size', self.batch_size)
            self.learning_rate = \
                hyperparameters.pop('learning_rate', self.learning_rate)
            self.kfold = \
                hyperparameters.pop('kfold', self.kfold)

            self.dropout = \
                hyperparameters.pop('dropout', self.dropout)
            self.embedding_dim = \
                hyperparameters.pop('embedding_dim', self.embedding_dim)
            self.hidden_dim = \
                hyperparameters.pop('hidden_dim', self.hidden_dim)
            self.stance_output_dim = \
                hyperparameters.pop('stance_output_dim', self.stance_output_dim)
            self.nli_output_dim = \
                hyperparameters.pop('nli_output_dim', self.nli_output_dim)
            self.num_rnn_layers = \
                hyperparameters.pop('num_rnn_layers', self.num_rnn_layers)
            self.num_linear_layers = \
                hyperparameters.pop('num_linear_layers', self.num_linear_layers)
            self.attention = \
                hyperparameters.pop('attention', self.attention)
            self.clip_grad_value = \
                hyperparameters.pop('clip_grad_value', self.clip_grad_value)
            self.nli_loss_weight = \
                hyperparameters.pop('nli_loss_weight', self.nli_loss_weight)
            self.lexicon_loss_weight = \
                hyperparameters.pop('lexicon_loss_weight', self.lexicon_loss_weight)

        return self

    def save_to_file(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        else:
            with open(file_path, 'wb') as f:
                hyperparameters = {
                    'stance_dataset': self.stance_dataset,
                    'embedding_file': self.embedding_file,
                    'lexicon_file': self.lexicon_file,
                    'random_seed': self.random_seed,
                    'epoch': self.epoch,
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'kfold': self.kfold,
                    'dropout': self.dropout,
                    'embedding_dim': self.embedding_dim,
                    'hidden_dim': self.hidden_dim,
                    'stance_output_dim': self.stance_output_dim,
                    'nli_output_dim': self.nli_output_dim,
                    'num_rnn_layers': self.num_rnn_layers,
                    'num_linear_layers': self.num_linear_layers,
                    'attention': self.attention,
                    'clip_grad_value': self.clip_grad_value,
                    'nli_loss_weight': self.nli_loss_weight,
                    'lexicon_loss_weight': self.lexicon_loss_weight
                }

                pickle.dump(hyperparameters, f)

        return self