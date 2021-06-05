# built-in modules
import argparse
import os
import json

class BaseConfig:
    def __init__(self):
        # construct argparser
        parser = argparse.ArgumentParser(
            description='Train the multi-task model'
        )

        # add argument to argparser

        # experiment_no
        parser.add_argument('--experiment_no',
                            default='1',
                            type=str)

        # evaluate setting
        parser.add_argument('--evaluate_nli',
                            default=0,
                            type=int)

        # model type
        parser.add_argument('--model',
                            default='task-specific-shared',
                            type=str)

        # preprocess
        parser.add_argument('--max_seq_len',
                            default=40,
                            type=int)

        # dataset
        parser.add_argument('--embedding',
                            default='wikipedia',
                            type=str)
        parser.add_argument('--nli_dataset_size',
                            default=0.1,
                            type=float)
        parser.add_argument('--stance_output_dim',
                            default=3,
                            type=int)
        parser.add_argument('--nli_output_dim',
                            default=3,
                            type=int)

        # hyperparameter
        parser.add_argument('--embedding_dim',
                            default=300,
                            type=int)

        parser.add_argument('--stance_hidden_dim',
                            default=128,
                            type=int)
        parser.add_argument('--nli_hidden_dim',
                            default=128,
                            type=int)
        parser.add_argument('--shared_hidden_dim',
                            default=128,
                            type=int)

        parser.add_argument('--stance_linear_dim',
                            default=100,
                            type=int)
        parser.add_argument('--nli_linear_dim',
                            default=100,
                            type=int)
        parser.add_argument('--shared_linear_dim',
                            default=100,
                            type=int)

        parser.add_argument('--num_stance_rnn',
                            default=1,
                            type=int)
        parser.add_argument('--num_nli_rnn',
                            default=1,
                            type=int)
        parser.add_argument('--num_shared_rnn',
                            default=1,
                            type=int)

        parser.add_argument('--num_stance_gcn',
                            default=1,
                            type=int)
        parser.add_argument('--num_nli_gcn',
                            default=1,
                            type=int)
        parser.add_argument('--num_shared_gcn',
                            default=1,
                            type=int)

        parser.add_argument('--num_stance_linear',
                            default=1,
                            type=int)
        parser.add_argument('--num_nli_linear',
                            default=1,
                            type=int)

        parser.add_argument('--attention',
                            default='linear',
                            type=str)
        parser.add_argument('--attention_threshold',
                            default=0.1,
                            type=float)

        parser.add_argument('--rnn_dropout',
                            default=0.3,
                            type=float)
        parser.add_argument('--gcn_dropout',
                            default=0.3,
                            type=float)
        parser.add_argument('--linear_dropout',
                            default=0.5,
                            type=float)

        parser.add_argument('--nli_loss_weight',
                            default=1.0,
                            type=float)

        parser.add_argument('--learning_rate',
                            default=1e-4,
                            type=float)
        parser.add_argument('--weight_decay',
                            default=1e-4,
                            type=float)
        parser.add_argument('--clip_grad_value',
                            default=1.0,
                            type=float)
        parser.add_argument('--lr_decay_step',
                            default=10,
                            type=int)
        parser.add_argument('--lr_decay',
                            default=1.0,
                            type=float)

        # other
        parser.add_argument('--random_seed',
                            default=77,
                            type=int)
        parser.add_argument('--kfold',
                            default=5,
                            type=int)
        parser.add_argument('--test_size',
                            default=0.2,
                            type=float)
        parser.add_argument('--epoch',
                            default=50,
                            type=int)
        parser.add_argument('--batch_size',
                            default=16,
                            type=int)

        self.parser = parser
        self.config = None

    def get_config(self):
        self.config = self.parser.parse_args()

    def save(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')

        # save config to json format
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False)

    def load(self, file_path=None):
        if file_path is None or type(file_path) != str:
            raise ValueError('argument `file_path` should be a string')
        elif not os.path.exists(file_path):
            raise FileNotFoundError('file {} does not exist'.format(file_path))

        # load config file
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.config = argparse.Namespace(**config)