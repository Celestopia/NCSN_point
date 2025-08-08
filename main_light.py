"""light-weight version of main.py"""
import os
import time
import numpy as np
import torch
import torch.optim as optim
import yaml
import json
import logging
import tqdm
import logging
import traceback

from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed
from utils.format import dict2namespace, namespace2dict
from utils.metrics import gmm_estimation, gmm_kl
from datasets.point import generate_point_dataset
from Langevin import PID_ALD

os.environ["OMP_NUM_THREADS"] = "5" # To avoid the warning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.


def main(args):
    
    try:
        
        # Logging
        if not os.path.exists(args.saving.result_dir):
            os.makedirs(args.saving.result_dir, exist_ok=True)
            print("Result directory created at {}.".format(args.saving.result_dir))
        
        time_string = str(int(time.time())) # Time string to identify the experiment
        experiment_dir = os.path.join(
                            args.saving.result_dir,
                            'experiment_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                time_string,
                                str(args.model.sigma_begin),
                                str(args.model.sigma_end),
                                str(args.model.num_classes),
                                str(args.sampling.n_steps_each),
                                str(args.sampling.k_p),
                                str(args.sampling.k_i),
                                str(args.sampling.k_d),
                                str(args.sampling.k_i_decay),
                                str(args.sampling.k_d_decay),
                                ))
        if args.saving.experiment_dir_suffix != '':
            experiment_dir += '_' + args.saving.experiment_dir_suffix
        elif args.saving.experiment_dir_suffix == '':
            experiment_dir += '_' + args.saving.experiment_name

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print("Experiment directory created at {}.".format(experiment_dir))

        from utils.log import get_logger, close_logger
        log_file_path = os.path.join(experiment_dir, 'log.log') # Set the log path
        logger = get_logger(log_file_path=log_file_path) # Set and get the root logger
        logging.info("Experiment directory: '{}'".format(experiment_dir))

        # Save the config file
        from utils.format import NumpyEncoder
        config_dict = namespace2dict(args)
        config_save_path = os.path.join(experiment_dir, 'config.json')
        json.dump(config_dict, open(config_save_path, 'w'), indent=4, cls=NumpyEncoder)
        logging.info("Experiment config saved to '{}'.".format(config_save_path))

    except Exception as e:
        print("Error: {}".format(e))
        return


    try: # Now the logger has been successfully set up, and errors can be logged in the log file.

        set_seed(args.seed)

        # Noise Scale Generation
        sigmas = torch.tensor(
                    np.exp(
                        np.linspace(
                            np.log(args.model.sigma_begin),
                            np.log(args.model.sigma_end),
                            args.model.num_classes
                        )
                    )
                ).float().to(args.device) # Shape: (num_classes,)
        sigmas_np = sigmas.cpu().numpy()

        # Model Configuration
        from models.simple_models import SimpleNet1d
        from utils import get_act

        used_activation = get_act(args.model.activation)
        score = SimpleNet1d(data_dim=2, hidden_dim=args.model.hidden_dim, sigmas=sigmas, act=used_activation).to(args.device)

        # Load model
        score.load_state_dict(torch.load(args.training.model_load_path), strict=True)
        logging.info("Model loaded from '{}'.".format(args.training.model_load_path))

        # Sampling
        gen = torch.Generator()
        gen.manual_seed(42) # Set the seed for random initial noise, so that it will be the same across different runs.
        initial_noise = (16*torch.rand(args.data.n_test_samples,2,generator=gen)-8).to('cpu') # uniformly sampled from [-8, 8]

        all_generated_samples = []
        all_generated_samples.append(initial_noise)

        from functools import partial
        sampler = partial(PID_ALD,
                        k_p=args.sampling.k_p,
                        k_i=args.sampling.k_i,
                        k_d=args.sampling.k_d,
                        k_i_window_size=args.sampling.k_i_window_size,
                        k_i_decay=args.sampling.k_i_decay,
                        k_d_decay=args.sampling.k_d_decay
                        )

        x_mods, sampler_record_dict = sampler(initial_noise.to(args.device), score, sigmas_np,
                        n_steps_each=args.sampling.n_steps_each,
                        step_lr=args.sampling.step_lr,
                        verbose=args.logging.sampling_verbose,)
        all_generated_samples.extend(x_mods)
        all_generated_samples = np.array([tensor.cpu().detach().numpy() for tensor in all_generated_samples]) # (num_classes * n_steps_each, n_test_samples, 2)
        logging.info("Generated samples shape: {}".format(all_generated_samples.shape))

        ## Evaluation
        frame_indices = np.linspace(1, len(all_generated_samples)-1, args.sampling.n_frames_each * args.model.num_classes + 1, dtype=int)

        logging.info("Start Evaluation...")
        kl_divergences, weights_preds, mu_preds, cov_preds = [], [], [], []

        weights_true = np.array(args.data.weights_true)
        mu_true = np.array(args.data.mu_true)
        cov_true = np.array(args.data.cov_true)
        #true_samples = generate_point_dataset(n_samples=1000, weights_true=weights_true, mu_true=mu_true, cov_true=cov_true) # (1000, 2)
        for t in tqdm.tqdm(frame_indices, desc='Evaluating...'):
            weights_pred, mu_pred, cov_pred = gmm_estimation(all_generated_samples[t], n_components=2)
            kl = gmm_kl(weights_true, mu_true, cov_true, weights_pred, mu_pred, cov_pred, n_samples=100000)
            kl_divergences.append(kl)
            mu_preds.append(mu_pred)
            cov_preds.append(cov_pred)
            weights_preds.append(weights_pred)

        kl_divergence_final = kl_divergences[-1]

        # Result saving
        logging.info("Saving Experiment Result...")
        result_dict = {
            'experiment_name': args.saving.experiment_name,
            'comment': args.saving.comment,
            'time_string': time_string,

            # Final Metric
            'kl_divergence_final': kl_divergence_final,

            # Parameters of the sampling process
            'weights_preds_final': weights_preds[-1].tolist(),
            'mu_preds_final': mu_preds[-1].tolist(),
            'cov_preds_final': cov_preds[-1].tolist(),

            # Metrics of each frame of the sampling process
            'kl_divergences': kl_divergences,

            # Parameters of each frame of the sampling process
            'weights_preds': [weights_pred.tolist() for weights_pred in weights_preds],
            'mu_preds': [mu_pred.tolist() for mu_pred in mu_preds],
            'cov_preds': [cov_pred.tolist() for cov_pred in cov_preds],
        }
        result_save_path = os.path.join(experiment_dir, 'result.json')
        json.dump(result_dict, open(result_save_path, 'w'), indent=4)
        logging.info("Experiment result saved to '{}'.".format(result_save_path))

        close_logger(logger)
        logging.info("Experiment finished.")

        return 0

    except Exception as e:
        logger.error("Error: {}".format(e))
        logger.error(traceback.format_exc())
        close_logger(logger)

        return e


if __name__ == '__main__':
    with open(os.path.join('configs', 'point_light.yml'), 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['saving']['result_dir'] = 'results111'
    args = dict2namespace(config_dict)
    main(args)

