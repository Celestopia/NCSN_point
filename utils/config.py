import torch
import numpy as np

hyperparameter_dict_default = { # version 20250328
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'model': {
        'sigma_begin': 32,
        'sigma_end': 0.01,
        'num_classes': 16,
        'activation': 'lrelu',
        'hidden_dim': 128,
    },
    'data': {
        'mu_true': np.array([[-3, -3],
                            [3, 3]]),
        'cov_true': np.array([[[1, 0],
                            [0, 1]],
                            [[1, 0],
                            [0, 1]]]),
        'weights_true': np.array([0.80, 0.20]),
        'n_train_samples': 100000,
        'n_test_samples': 100,
    },
    'training': {
        'batch_size': 128,
        'n_epochs': 20,
    },
    'sampling': {
        'sampler': 'ald', # options: ['ald', 'fcald']
        'batch_size': 64, # TODO: 暂时没用到
        'n_steps_each': 50,
        'step_lr': 0.000008,
        'k_p': 1.0,
        'k_i': 0.0,
        'k_d': 0.0,
    },
    'optim': {
        'optimizer': 'Adam',
        'lr': 0.0001,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.000,
    },
    'visualization': {
        'n_frames': 200,
        'figsize': (12,12),
    },
    'saving': {
        'result_dir': 'results',
        'experiment_dir_suffix': '',
        'experiment_name': 'default',
        'comment': '',
    },
}
