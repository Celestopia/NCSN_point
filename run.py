import numpy as np
import os
import sys
from main import main
from utils.format import dict2namespace, set_namespace_value
from utils.config import hyperparameter_dict_default


def run(args, keys_list, values_list, result_dir):
    """
    Run multiple experiments with different hyperparameters.
    
    Args:
        args: argparse.Namespace, the default hyperparameters
        keys_list (list of list): The keys of hyperparameters to be changed
        values_list (list of list): The values of hyperparameters to be changed
        result_dir (str): The directory to save the results
    
    Examples:
    - 
        ```python
        keys_list = [['model', 'activation']]
        values_list = [['relu','elu','lrelu','selu','gelu','silu','swish','mish','sigmoid','tanh']]
        result_dir = 'results_activation' 
        run(args, keys_list, values_list, result_dir)
        ```
    - 
        ```python
        keys_list = [['training', 'n_epochs'],['data', 'n_train_samples']]
        values_list = [[200,80,40,20,10,5],[10000,25000,50000,100000,200000,400000]]
        result_dir = 'results_n_train_samples_n_epochs' 
        run(args, keys_list, values_list, result_dir)
        ```
    """
    args.saving.result_dir = result_dir
    num_exp = len(values_list[0])
    for i in range(num_exp):
        experiment_name = '_'.join([f'{keys[-1]}={values[i]}' for keys, values in zip(keys_list, values_list)])
        experiment_dir_suffix = '_'.join([f'{keys[-1]}={values[i]}' for keys, values in zip(keys_list, values_list)])
        for keys, values in zip(keys_list, values_list):
            set_namespace_value(args, keys, values[i])
        args.saving.experiment_name = experiment_name
        args.saving.experiment_dir_suffix = experiment_dir_suffix
        main(args)


args=dict2namespace(hyperparameter_dict_default)
keys_list = [['seed']]
values_list = [[0,1,12,123,1234,12345,123456,1234567,12345678]]
result_dir = 'results_seed'
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
keys_list = [['model', 'activation']]
values_list = [['relu','elu','lrelu','selu','gelu','silu','swish','mish','sigmoid','tanh', 'softplus']]
result_dir = 'results_activation' 
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
keys_list = [['training', 'n_epochs']]
values_list = [[5,10,15,20,30,40,50,60,80,100]]
result_dir = 'results_n_epochs'
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
keys_list = [['data', 'n_train_samples']]
values_list = [[1000,5000,20000,100000,250000,1000000]]
result_dir = 'results_n_train_samples'
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
keys_list = [['sampling', 'n_steps_each']]
values_list = [[10,20,30,40,50,75,100,150,200]]
result_dir = 'results_n_steps_each' 
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
keys_list = [['sampling', 'step_lr']]
values_list = [[0.000002,0.000004,0.000006,0.000008,0.000010,0.000012,0.000014,0.00016]]
result_dir = 'results_step_lr' 
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
args.sampler = 'fcald'
keys_list = [['sampling', 'k_p']]
values_list = [[0.1,0.3,0.5,1.0,1.5,2.0,3.0,5.0]]
result_dir = 'results_k_p'
run(args, keys_list, values_list, result_dir)
args=dict2namespace(hyperparameter_dict_default)
args.sampler = 'fcald'
keys_list = [['sampling', 'k_i']]
values_list = [[0.0,0.01,0.03,0.05,0.10,0.30,0.50,1.00,3.00]]
result_dir = 'results_k_i'
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
args.sampler = 'fcald'
keys_list = [['sampling', 'k_d']]
values_list = [[0.0,0.0001,0.0003,0.0005,0.001,0.003,0.005,0.010,0.030,0.050]]
result_dir = 'results_k_d'
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
args.sampler = 'fcald'
keys_list = [['sampling', 'k_i'], ['sampling', 'k_d']]
values_list = [[0.0,0.01,0.03,0.05,0.10,0.30,0.50,1.00,3.00], [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]]
result_dir = 'results_k_d=0.001_k_i'
run(args, keys_list, values_list, result_dir)


args=dict2namespace(hyperparameter_dict_default)
args.sampler = 'fcald'
keys_list = [['sampling', 'k_i'], ['sampling', 'k_d']]
values_list = [[0.0,0.01,0.03,0.05,0.10,0.30,0.50,1.00,3.00], [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]]
result_dir = 'results_k_d=0.01_k_i'
run(args, keys_list, values_list, result_dir)

