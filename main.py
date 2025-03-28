import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import time
import argparse
import random
import logging
import copy
import tqdm
import logging
import traceback
import json
from torch.utils.data import DataLoader, Dataset
from utils import set_seed
from utils.format import dict2namespace, namespace2dict
matplotlib.use('Agg') # Set the backend to disable figure display window



hyperparameter_dict = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'model': {
        'sigma_begin': 32,
        'sigma_end': 0.01,
        'num_classes': 36,
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
        'n_steps_each': 25,
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
        'result_dir': 'results121',
        'experiment_dir_suffix': '',
        'experiment_name': 'default',
        'comment': '',
    },
}
args=dict2namespace(hyperparameter_dict)



def main(args):
    
    try:
        
        # Logging
        if not os.path.exists(args.saving.result_dir):
            os.makedirs(args.saving.result_dir)
            print("Result directory created at {}.".format(args.saving.result_dir))
        
        time_string = str(int(time.time()))
        experiment_dir = os.path.join(
                            args.saving.result_dir,
                            'experiment_{}_{}_{}_{}_{}'.format(
                                time_string,
                                str(args.model.sigma_begin),
                                str(args.model.sigma_end),
                                str(args.model.num_classes),
                                str(args.sampling.n_steps_each)
                                ))
        if args.saving.experiment_dir_suffix:
            experiment_dir += '_' + args.saving.experiment_dir_suffix

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print("Experiment directory created at {}.".format(experiment_dir))

        from utils.log import get_logger, close_logger
        log_file_path = os.path.join(experiment_dir, 'log.log') # Set the log path
        logger = get_logger(log_file_path=log_file_path)

    except Exception as e:
        print("Error: {}".format(e))
        return


    try: # Now the logger has been successfully set up, and errors can be logged in the log file.

        set_seed(args.seed)

        # Data Preparation
        from datasets.point import generate_point_dataset, PointDataset

        data = generate_point_dataset(n_samples=args.data.n_train_samples, mu_true=args.data.mu_true, cov_true=args.data.cov_true, weights_true=args.data.weights_true)
        data = torch.tensor(data, dtype=torch.float32).to(args.device)
        #data_copy = data.clone()
        logger.info("Training data shape: {}".format(data.shape))

        train_dataset = PointDataset(data)
        train_loader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=0)

        #test_dataset = PointDataset(data[:100]) # generative task, no need for test set
        #test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        sigmas = torch.tensor(
                    np.exp(
                        np.linspace(
                            np.log(args.model.sigma_begin),
                            np.log(args.model.sigma_end),
                            args.model.num_classes
                        )
                    )
                ).float().to(args.device) # (num_classes,)


        # Model Configuration
        from models.simple_models import SimpleNet1d, SimpleResNet
        from utils import get_act
        used_activation = get_act(args.model.activation)
        score = SimpleNet1d(data_dim=2, hidden_dim=args.model.hidden_dim, sigmas=sigmas, act=used_activation).to(args.device)
        optimizer = optim.Adam(score.parameters(), lr=args.optim.lr, weight_decay=args.optim.weight_decay, betas=(args.optim.beta1, args.optim.beta2), eps=args.optim.eps)


        # Training
        from Langevin import anneal_dsm_score_estimation

        score.train()
        step=0
        for epoch in tqdm.tqdm(range(args.training.n_epochs), desc='Training...'):
            for i, X in enumerate(train_loader):
                step += 1
                X = X.to(args.device)

                loss = anneal_dsm_score_estimation(score, X, sigmas)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 100 == 0:
                    logger.info("epoch: {}, step: {}, loss: {}".format(epoch, step, loss.item()))


        # Sampling
        from Langevin import anneal_Langevin_dynamics, FC_ALD

        initial_noise=2*torch.randn(args.data.n_test_samples,2).to('cpu')
        all_generated_samples = [initial_noise]

        if args.sampling.sampler == 'ald':
            all_generated_samples.extend(anneal_Langevin_dynamics(initial_noise.to(args.device), score, sigmas,
                                                                n_steps_each=args.sampling.n_steps_each,
                                                                step_lr=args.sampling.step_lr,
                                                                verbose=False
                                                            ))
        elif args.sampling.sampler == 'fcald':
            all_generated_samples.extend(FC_ALD(initial_noise.to(args.device), score, sigmas,
                                                    n_steps_each=args.sampling.n_steps_each,
                                                    step_lr=args.sampling.step_lr,
                                                    k_p=args.sampling.k_p,
                                                    k_i=args.sampling.k_i,
                                                    k_d=args.sampling.k_d,
                                                    verbose=False
                                                ))
        all_generated_samples = np.array([tensor.cpu().detach().numpy() for tensor in all_generated_samples]) # (num_classes * n_steps_each, args.data.n_test_samples, 2)
        logger.info("Generated samples shape: {}".format(all_generated_samples.shape))


        # Evaluation
        from utils.measure import sample_wasserstein_distance, gmm_estimation, sample_mmd2_rbf, kl_gmms

        frame_indices = np.linspace(0, len(all_generated_samples)-1, args.visualization.n_frames, dtype=int)
        frame_samples = all_generated_samples[frame_indices] # Select some samples for evaluation, and for animation frames
        logger.info("Frame samples shape: {}".format(frame_samples.shape)) # (n_frames, args.data.n_test_samples, 2)
        true_samples = generate_point_dataset(n_samples=1000, mu_true=args.data.mu_true, cov_true=args.data.cov_true, weights_true=args.data.weights_true) # (1000, 2)

        mu_preds, cov_preds, weights_preds = [], [], []
        for t in tqdm.tqdm(frame_indices, desc='Evaluating GMM...'):
            generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
            mu_pred, cov_pred, weights_pred = gmm_estimation(generated_samples)
            mu_preds.append(mu_pred)
            cov_preds.append(cov_pred)
            weights_preds.append(weights_pred)

        kl_divergences = []
        for i, t in tqdm.tqdm(enumerate(frame_indices), desc='Evaluating KL divergence...'):
            generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
            kl_divergences.append(kl_gmms(args.data.weights_true, args.data.mu_true, args.data.cov_true,
                                                weights_preds[i], mu_preds[i], cov_preds[i], n_samples=10000))

        mmd2s = []
        for t in tqdm.tqdm(frame_indices, desc='Evaluating MMD...'):
            generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
            mmd2s.append(sample_mmd2_rbf(generated_samples, true_samples))

        wasserstein_distances = []
        for t in tqdm.tqdm(frame_indices, desc='Evaluating Wasserstein Distance...'):
            generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
            wasserstein_distances.append(sample_wasserstein_distance(generated_samples, true_samples))

        print("Final KL divergence: ", kl_divergences[-1])
        print("Final MMD2: ", mmd2s[-1])
        print("Final Wasserstein Distance: ", wasserstein_distances[-1])
        print("Predicted GMM parameters: \n", mu_pred, "\n", cov_pred, "\n", weights_pred)

        # Visualization (static)
        plt.figure(figsize=args.visualization.figsize)
        for i, f_idx in enumerate(np.linspace(0, len(frame_indices)-1, 16, dtype=int)):
            # frame_indices is the list of indices selected for evaluation and visualization. e.g. frame_indices = [0, 20, 40, 60, ..., 2000].
            # f_idx is the index of a frame in `frame_indices`. e.g. f_idx=2, and the corresponding t is frame_indices[f_idx]=40.
            t=frame_indices[f_idx]
            plt.subplot(4, 4, i+1)
            plt.title(f"t={t}")
            plt.text(0.02, 0.95, "KL:   {:+.3e}".format(kl_divergences[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
            plt.text(0.02, 0.90, "MMD2: {:+.3e}".format(mmd2s[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
            plt.text(0.02, 0.85, "WD:   {:+.3e}".format(wasserstein_distances[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
            plt.scatter(frame_samples[f_idx][:, 0], frame_samples[f_idx][:, 1], s=2)
            plt.scatter([args.data.mu_true[0][0]], [args.data.mu_true[0][1]], s=50, c='r', marker='+')
            plt.scatter([args.data.mu_true[1][0]], [args.data.mu_true[1][1]], s=50, c='g', marker='+')
            plt.scatter([mu_preds[f_idx][0][0]], [mu_preds[f_idx][0][1]], s=10, c='r')
            plt.scatter([mu_preds[f_idx][1][0]], [mu_preds[f_idx][1][1]], s=10, c='g')
        fig_save_path = os.path.join(experiment_dir,'4x4_visualization.png')
        plt.tight_layout() # Adjust subplot spacing to avoid overlap
        plt.savefig(fig_save_path, dpi=300)
        logger.info("Figure saved to '{}'.".format(fig_save_path))
        plt.show()


        # Visualization (animation)
        from utils.animation import make_point_animation

        fig, ax = plt.subplots(figsize=args.visualization.figsize)
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        frame_text_func = lambda frame: "Frame {}/{}\nKL divergence: {:.8f}\nMMD2: {:.8f}\nWasserstein Distance: {:.8f}".format(
                                                                                                    frame,
                                                                                                    len(frame_samples),
                                                                                                    kl_divergences[frame],
                                                                                                    mmd2s[frame],
                                                                                                    wasserstein_distances[frame],
                                                                                                )
        ani = make_point_animation(fig, ax, frame_samples, frame_text_func=frame_text_func)
        ax.scatter([args.data.mu_true[0][0]], [args.data.mu_true[0][1]], s=50, c='r', marker='+')
        ax.scatter([args.data.mu_true[1][0]], [args.data.mu_true[1][1]], s=50, c='g', marker='+')
        animation_save_path = os.path.join(experiment_dir,'animation.gif')
        ani.save(animation_save_path, writer='pillow', fps=30) # Save animation as gif
        logger.info("Animation saved to '{}'.".format(animation_save_path))
        plt.show()



        # Result saving
        from utils.format import NumpyEncoder

        result_dict = {
            'experiment_name': args.saving.experiment_name,
            'comment': args.saving.comment,
            'time_string': time_string,
            'kl_divergence_final': kl_divergences[-1],
            'mmd2_rbf_final': mmd2s[-1],
            'wasserstein_distance_final': wasserstein_distances[-1],
            'mu_preds_final': mu_preds[-1].tolist(),
            'cov_preds_final': cov_preds[-1].tolist(),
            'weights_preds_final': weights_preds[-1].tolist(),
            'kl_divergences': kl_divergences,
            'mmd2s': mmd2s,
            'wasserstein_distances': wasserstein_distances,
            'mu_preds': [mu_pred.tolist() for mu_pred in mu_preds],
            'cov_preds': [cov_pred.tolist() for cov_pred in cov_preds],
            'weights_preds': [weights_pred.tolist() for weights_pred in weights_preds],
        }
        result_save_path = os.path.join(experiment_dir, 'result.json')
        json.dump(result_dict, open(result_save_path, 'w'), indent=4)
        logger.info("Experiment result saved to '{}'.".format(result_save_path))

        config_dict = namespace2dict(args)
        config_save_path = os.path.join(experiment_dir, 'config.json')
        json.dump(config_dict, open(config_save_path, 'w'), indent=4, cls=NumpyEncoder)
        logger.info("Experiment config saved to '{}'.".format(config_save_path))

        close_logger(logger)

        return 0

    except Exception as e:
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        close_logger(logger)

        return e


if __name__ == '__main__':
    main(args)

