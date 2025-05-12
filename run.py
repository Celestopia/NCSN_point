import os
import sys
import numpy as np
import argparse
from main import main
from utils.format import get_leaf_nodes, compare_dicts, namespace2dict, dict2namespace, set_namespace_value
from configs.point import hyperparameter_dict_default



class Exp:
    """A class for running multiple experiments with different hyperparameters."""
    def __init__(self, main, hyperparameter_dict_default):
        self.main = main
        self.hyperparameter_dict_default = hyperparameter_dict_default
        self.kvs_default = get_leaf_nodes(hyperparameter_dict_default) # kvs: key-value pairs

    def run(self, input):
        if isinstance(input, dict):
            hyperparameter_dict = input
        elif isinstance(input, argparse.Namespace):
            hyperparameter_dict = namespace2dict(input)
        else:
            raise ValueError('Input must be a dictionary or an argparse.Namespace')
        kvs = get_leaf_nodes(hyperparameter_dict)
        diff_kvs = compare_dicts(self.kvs_default, kvs) # different key-value pairs
        experiment_name = '_'.join([f'{k.split('.')[-1]}={v}' for k, v in diff_kvs.items()])
        
        args = dict2namespace(hyperparameter_dict)
        args.saving.experiment_name = experiment_name
        args.saving.experiment_dir_suffix = experiment_name
        print(f'experiment_name: {experiment_name}')
        self.main(args)

    def line_run(self, key, values, fixed_kv_pairs=[]):
        """
        Run multiple experiments changing one hyperparameter.

        Args:
            key (str): Dot-separated string, name of the hyperparameter to be changed, e.g. 'data.batch_size'.
            values (list): Values of the hyperparameter to be tested.
            fixed_kv_pairs (list): Other key-value pairs to be modified and then fixed.
        """
        args = dict2namespace(self.hyperparameter_dict_default)
        for k, v in fixed_kv_pairs:
            set_namespace_value(args, k, v)
        for value in values:
            set_namespace_value(args, key, value)
            self.run(args)

    def grid_run(self, key1, values1, key2, values2, fixed_kv_pairs=[]):
        """
        Run multiple experiments changing two hyperparameters.

        Args:
            key1 (str): Dot-separated string, name of the first hyperparameter to be changed, e.g. 'data.batch_size'.
            values1 (list): Values of the first hyperparameter to be tested.
            key2 (str): Dot-separated string, name of the second hyperparameter to be changed, e.g. 'training.lr'.
            values2 (list): Values of the second hyperparameter to be tested.
            fixed_kv_pairs (list of tuples): Other key-value pairs to be modified and then fixed, e.g. [('data.dataset', 'cifar10'), ('optimizer.lr', 0.001)].
        """
        args = dict2namespace(self.hyperparameter_dict_default)
        for k, v in fixed_kv_pairs:
            set_namespace_value(args, k, v)
        for value1 in values1:
            for value2 in values2:
                set_namespace_value(args, key1, value1)
                set_namespace_value(args, key2, value2)
                self.run(args)


hyperparameter_dict_default['saving']['result_dir']='results128'
hyperparameter_dict_default['seed']=76923

exp = Exp(main, hyperparameter_dict_default)
exp.line_run('sampling.k_d', [10.0,11.0,12.0], fixed_kv_pairs=[('sampling.k_i', 0.1), ('sampling.k_i_decay', 0.9)])







