import torch
import numpy as np

hyperparameter_dict_default = { # Version 20250410
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'model': {
        'model_name': 'mlp', # options: ['mlp', 'resnet']
        'sigma_begin': 20,
        'sigma_end': 0.01,
        'num_classes': 8,
        'activation': 'softplus',
        'hidden_dim': 128,
    },
    'data': {
        'weights_true': np.array([0.80, 0.20]),
        'mu_true': np.array([[5, 5],
                            [-5, -5]]),
        'cov_true': np.array([[[1, 0],
                            [0, 1]],
                            [[1, 0],
                            [0, 1]]]),
        'n_train_samples': 100000,
        'n_test_samples': 1280,
    },
    'training': {
        'batch_size': 128,
        'n_epochs': 10,
        'model_load_path': None, # If not None, load the model from the specified path.
    },
    'sampling': {
        'sampler': 'fcald', # options: ['ald', 'fcald']
        'batch_size': 64, # TODO: 暂时没用到
        'n_steps_each': 150,
        'step_lr': 8e-6,
        'k_p': 1.0,
        'k_i': 0.0,
        'k_d': 0.0,
    },
    'optim': {
        'optimizer': 'adam',
        'lr': 0.0001,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.000,
    },
    'visualization': {
        'n_frames_each': 30,
        'figsize': (12,12),
        'trajectory_start_point': [[1,1]],
    },
    'saving': {
        'result_dir': 'results',
        'experiment_dir_suffix': '',
        'experiment_name': 'default',
        'comment': '',
        'save_model': False,
        'save_figure': True,
        'save_animation': False,
        'save_trajectory': True,
        'save_metric_plot': True,
    },
}