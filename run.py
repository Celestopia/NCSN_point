import numpy as np
import os
import sys
from main import dict2namespace, main
from utils.config import hyperparameter_dict_default


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_sigma_begin'
for i in [0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0]:
    args.saving.experiment_name = 'sigma_begin={}'.format(i)
    args.saving.comment = ''

    args.model.sigma_begin = i
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_sigma_end'
for i in [0.05,0.03,0.01,0.005,0.003,0.001]:
    args.saving.experiment_name = 'sigma_end={}'.format(i)
    args.saving.comment = ''

    args.model.sigma_end = i
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_activation'
for i in ['relu','elu','lrelu','selu','gelu','silu','swish','mish','sigmoid','tanh']:
    args.saving.experiment_name = 'activation={}'.format(i)
    args.saving.experiment_dir_suffix = i
    args.saving.comment = ''

    args.model.activation = i
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_num_classes_n_steps_each'
for i, j in zip([8,12,16,24,30,40,60], [150,100,75,50,40,30,20]):
    args.saving.experiment_name = 'num_classes={}, n_steps_each={}'.format(i, j)
    args.saving.comment = ''

    args.model.num_classes = i
    args.sampling.n_steps_each = j
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_n_train_samples_n_epochs'
for i, j in zip([10000,25000,50000,100000,200000,400000], [200,80,40,20,10,5]):
    args.saving.experiment_name = 'n_train_samples={}, n_epochs={}'.format(i, j)
    args.saving.experiment_dir_suffix = 'n_train_samples={}_n_epochs={}'.format(i, j)
    args.saving.comment = ''

    args.data.n_train_samples = i
    args.training.n_epochs = j
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_seed'
for i in [0,42,123,2025,65536,9765625,20250328]:
    args.saving.experiment_name = 'seed={}'.format(i)
    args.saving.experiment_dir_suffix = 'seed={}'.format(i)
    args.saving.comment = ''

    args.seed = i
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_hidden_dim'
for i in [64,128,192,256,384,512]:
    args.saving.experiment_name = 'hidden_dim={}'.format(i)
    args.saving.experiment_dir_suffix = 'hidden_dim={}'.format(i)
    args.saving.comment = ''

    args.model.hidden_dim = i
    main(args)


args=dict2namespace(hyperparameter_dict_default)
args.saving.result_dir = 'results_1000_test_seed'
args.data.n_test_samples = 1000
for i in [0,42,123,2025,3473,65536,9765625,20250328]:
    args.saving.experiment_name = 'n_test_samples=1000, seed={}'.format(i)
    args.saving.experiment_dir_suffix = 'n_test_samples=1000_seed={}'.format(i)
    args.saving.comment = ''

    args.seed = i
    main(args)
