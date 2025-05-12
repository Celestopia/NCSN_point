"""Default hyperparameters for 2-d point dataset."""
import torch
import numpy as np

hyperparameter_dict_default = {
    'device': 'cuda',
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
        'weights_true': [0.80, 0.20],
        'mu_true': [[5, 5], [-5, -5]],
        'cov_true': [[[1, 0], [0, 1]],
                    [[1, 0], [0, 1]]],
        'n_train_samples': 100000,
        'n_test_samples': 1280,
    },
    'training': {
        'batch_size': 128,
        'n_epochs': 60,
        'train': False,
        'model_load_path': r"model_weights\scorenet_20_0.01_8.pth", # If not None, load the model from the specified path.
    },
    'sampling': {
        'sampler': 'pidald', # options: ['ald', 'pidald']
        'batch_size': 64, # TODO: not used currently
        'n_steps_each': 150,
        'step_lr': 8e-6,
        'k_p': 1.0,
        'k_i': 0.0,
        'k_d': 0.0,
        'k_i_window_size': 150,
        'k_i_decay': 1.00,
        'k_d_decay': 1.00,
    },
    'logging': {
        'training_log_freq': 100,
        'sampling_verbose': True,
        'sampling_log_freq': 1,
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
        'n_frames_each': 30, # Number of selected frames at each noise level, where metrics would be calculated.
        'figsize': (12,12),
        'trajectory_start_point': [[1,1]], # Starting point of the trajectory. Effective when args.saving.save_trajectory is True.
    },
    'saving': {
        'result_dir': 'results12345',
        'experiment_dir_suffix': '',
        'experiment_name': 'default',
        'comment': '',
        'save_model': True,
        'save_sample': True,
        'save_figure': True,
        'save_animation': False,
        'save_trajectory': False,
        'save_generation_metric_plot': True,
        'save_sampler_record_plot': True,
    },
}