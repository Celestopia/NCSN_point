import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
from utils.metrics import sample_wasserstein_distance, gmm_estimation, sample_mmd2_rbf, gmm_kl, gmm_log_likelihood
from utils.format import dict2namespace, namespace2dict
from datasets.point import generate_point_dataset
from Langevin import anneal_dsm_score_estimation, PID_ALD


sns.set_theme(style="whitegrid")
matplotlib.use('Agg') # Set the backend to disable figure display window
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
        optimizer = optim.Adam(score.parameters(), lr=args.optim.lr,
                                weight_decay=args.optim.weight_decay, betas=(args.optim.beta1, args.optim.beta2), eps=args.optim.eps)


        # Training
        if args.training.model_load_path is not None:
            score.load_state_dict(torch.load(args.training.model_load_path))
            logging.info("Model loaded from '{}'.".format(args.training.model_load_path))

        if args.training.train:
            logging.info(">>>>>>>>>>>>>>>>>>>> Model Training Started >>>>>>>>>>>>>>>>>>>>")
            
            # Data Preparation
            data = generate_point_dataset(n_samples=args.data.n_train_samples,
                                        weights_true=np.array(args.data.weights_true),
                                        mu_true=np.array(args.data.mu_true),
                                        cov_true=np.array(args.data.cov_true))
            data = torch.tensor(data, dtype=torch.float32).to(args.device) # Shape: (n_train_samples, 2)
            logging.info("Training data shape: {}".format(data.shape))
            train_loader = DataLoader(TensorDataset(data), batch_size=args.training.batch_size, shuffle=True, num_workers=args.data.num_workers)

            score.train()
            step=0
            for epoch in tqdm.tqdm(range(args.training.n_epochs), desc='Training...'):
                for i, X in enumerate(train_loader):
                    X = X.to(args.device)
                    loss = anneal_dsm_score_estimation(score, X, sigmas)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if step % args.logging.training_log_freq == 0:
                        logging.info("epoch: {}, step: {}, loss: {}".format(epoch, step, loss.item()))
                    step += 1
            if args.saving.save_model:
                model_save_path = os.path.join(experiment_dir,'scorenet_{}_{}_{}.pth'.format(
                                        args.model.sigma_begin, args.model.sigma_end, args.model.num_classes))
                torch.save(score.state_dict(), model_save_path)
                logging.info("Model saved to {}.".format(model_save_path))
            logging.info("<<<<<<<<<<<<<<<<<<<< Model Training Finished <<<<<<<<<<<<<<<<<<<<")


        # Sampling
        gen = torch.Generator()
        gen.manual_seed(42) # Set the seed for random initial noise, so that it will be the same across different runs.
        initial_noise = (16*torch.rand(args.data.n_test_samples,2,generator=gen)-8).to('cpu') # uniformly sampled from [-8, 8]

        if args.saving.save_trajectory: # If save_trajectory is True, add a point as the first instance of initial points, and visualize its trajectory later.
            trajectory_start_point = torch.tensor(args.visualization.trajectory_start_point) # (1,2)
            initial_noise = torch.concat([trajectory_start_point, initial_noise], dim=0) # (n_test_samples+1, 2)

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
        if args.saving.save_sample:
            sample_save_path = os.path.join(experiment_dir, 'all_generated_samples.npy')
            np.save(sample_save_path, all_generated_samples)
            logging.info("Generated samples saved to '{}'.".format(sample_save_path))
        del x_mods # Free memory

        # Evaluation
        def evaluate(samples, weights_true, mu_true, cov_true, n_true_samples=1000):
            """Evaluate the samples at some step using multiple metrics."""
            # samples: (n_samples, 2)
            true_samples = generate_point_dataset(n_samples=n_true_samples, weights_true=weights_true, mu_true=mu_true, cov_true=cov_true) # (1000, 2)
            weights_pred, mu_pred, cov_pred = gmm_estimation(samples)
            kl = gmm_kl(weights_true, mu_true, cov_true, weights_pred, mu_pred, cov_pred, n_samples=100000)
            log_likelihood = gmm_log_likelihood(samples, weights_pred, mu_pred, cov_pred)
            mmd2 = sample_mmd2_rbf(samples, true_samples)
            wasserstein_distance = sample_wasserstein_distance(samples, true_samples)
            return kl, log_likelihood, mmd2, wasserstein_distance, mu_pred, cov_pred, weights_pred

        ## Evaluation - metrics of each frame
        frame_indices = np.linspace(1, len(all_generated_samples)-1, args.sampling.n_frames_each * args.model.num_classes + 1, dtype=int)
        frame_samples = all_generated_samples[frame_indices] # Select some samples for evaluation and for animation frames
        logging.info("Frame samples shape: {}".format(frame_samples.shape)) # (n_frames, n_test_samples, 2)

        logging.info("Start Evaluation...")
        kl_divergences, log_likelihoods, mmd2s, wasserstein_distances, weights_preds, mu_preds, cov_preds = [], [], [], [], [], [], []
        for t in tqdm.tqdm(frame_indices, desc='Evaluating...'):
            kl, log_likelihood, mmd2, wasserstein_distance, mu_pred, cov_pred, weights_pred = \
                evaluate(all_generated_samples[t], np.array(args.data.weights_true), np.array(args.data.mu_true), np.array(args.data.cov_true))
            kl_divergences.append(kl)
            log_likelihoods.append(log_likelihood)
            mmd2s.append(mmd2)
            wasserstein_distances.append(wasserstein_distance)
            mu_preds.append(mu_pred)
            cov_preds.append(cov_pred)
            weights_preds.append(weights_pred)

        ## Evaluation - metrics of final samples (with different seeds during sampling)
        kl_divergence_finals = [] + [kl_divergences[-1]]
        log_likelihood_finals = [] + [log_likelihoods[-1]]
        mmd2_finals = [] + [mmd2s[-1]]
        wasserstein_distance_finals = [] + [wasserstein_distances[-1]]

        for sampling_seed in [76923,
                              153846,230769,307692,384615,461538,538461,615384,692307,769230,846153,923076,1000000
                            ]: # Run sampling process again with different seeds to quantify the variation of metrics
            set_seed(sampling_seed)
            reconstructed_samples, _ = sampler(initial_noise.to(args.device), score, sigmas_np,
                                                            n_steps_each=args.sampling.n_steps_each,
                                                            step_lr=args.sampling.step_lr,
                                                            verbose=False,
                                                            final_only=True) # Note final_only=True
            reconstructed_samples = reconstructed_samples[-1].cpu().detach().numpy() # (n_test_samples, 2)
            kl, log_likelihood, mmd2, wasserstein_distance, _, _, _ = \
                evaluate(reconstructed_samples, np.array(args.data.weights_true), np.array(args.data.mu_true), np.array(args.data.cov_true))
            kl_divergence_finals.append(kl)
            log_likelihood_finals.append(log_likelihood)
            mmd2_finals.append(mmd2)
            wasserstein_distance_finals.append(wasserstein_distance)
            del reconstructed_samples # Free memory

        kl_divergence_final, kl_divergence_final_std = np.array(kl_divergence_finals).mean(), np.array(kl_divergence_finals).std()
        log_likelihood_final, log_likelihood_final_std = np.array(log_likelihood_finals).mean(), np.array(log_likelihood_finals).std()
        mmd2_final, mmd2_final_std = np.array(mmd2_finals).mean(), np.array(mmd2_finals).std()
        wasserstein_distance_final, wasserstein_distance_final_std = np.array(wasserstein_distance_finals).mean(), np.array(wasserstein_distance_finals).std()

        logging.info("Final KL divergence: {} +- 3 * {}".format(kl_divergence_final, kl_divergence_final_std))
        logging.info("Final Log likelihood: {} +- 3 * {}".format(log_likelihood_final, log_likelihood_final_std))
        logging.info("Final MMD2: {} +- 3 * {}".format(mmd2_final, mmd2_final_std))
        logging.info("Final Wasserstein Distance: {} +- 3 * {}".format(wasserstein_distance_final, wasserstein_distance_final_std))


        # Visualization
        ## Visualization - static
        if args.saving.save_figure:
            logging.info("Saving Point Visualization...")
            plt.figure(figsize=args.visualization.figsize) # Figure of generated samples at different stages of sampling
            for i, f_idx in enumerate(np.linspace(0, len(frame_indices)-1, 16, dtype=int)):
                # frame_indices is the list of indices selected for evaluation and visualization. e.g. frame_indices = [0, 20, 40, 60, ..., 2000].
                # f_idx is the index of a frame in `frame_indices`. e.g. f_idx=2, and the corresponding t is frame_indices[f_idx]=40.
                t=frame_indices[f_idx]
                plt.subplot(4, 4, i+1)
                plt.title(f"t={t}")
                plt.text(0.02, 0.95, "KL:   {:+.3e}".format(kl_divergences[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
                plt.text(0.02, 0.90, "LL:   {:+.3e}".format(log_likelihoods[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
                plt.text(0.02, 0.85, "MMD2: {:+.3e}".format(mmd2s[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
                plt.text(0.02, 0.80, "WD:   {:+.3e}".format(wasserstein_distances[f_idx]), fontsize=7, transform=plt.gca().transAxes, fontfamily='consolas')
                plt.scatter(frame_samples[f_idx][:, 0], frame_samples[f_idx][:, 1], s=0.5)
                plt.scatter([args.data.mu_true[0][0]], [args.data.mu_true[0][1]], s=50, c='r', marker='+') # The ideal center of the first cluster
                plt.scatter([args.data.mu_true[1][0]], [args.data.mu_true[1][1]], s=50, c='g', marker='+') # The ideal center of the second cluster
                plt.scatter([mu_preds[f_idx][0][0]], [mu_preds[f_idx][0][1]], s=10, c='r') # The true center of the first cluster
                plt.scatter([mu_preds[f_idx][1][0]], [mu_preds[f_idx][1][1]], s=10, c='g') # The true center of the second cluster
            fig_save_path = os.path.join(experiment_dir,'sample_plot.png')
            plt.tight_layout() # Adjust subplot spacing to avoid overlap
            plt.savefig(fig_save_path, dpi=300, bbox_inches='tight')
            logging.info("Figure saved to '{}'.".format(fig_save_path))
            plt.close()


        ## Visualization - animation
        if args.saving.save_animation:
            logging.info("Saving Point Animation...")
            from utils.animation import make_point_animation

            fig, ax = plt.subplots(figsize=args.visualization.figsize)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            frame_text_func = lambda frame: "Frame {}/{}\nKL divergence: {:.8f}\nLog likelihood: {:.8f}\nMMD2: {:.8f}\nWasserstein Distance: {:.8f}".format(
                                        frame+1, len(frame_samples), kl_divergences[frame], log_likelihoods[frame], mmd2s[frame], wasserstein_distances[frame])
            ani = make_point_animation(fig, ax, frame_samples, frame_text_func=frame_text_func)
            ax.scatter([args.data.mu_true[0][0]], [args.data.mu_true[0][1]], s=50, c='r', marker='+')
            ax.scatter([args.data.mu_true[1][0]], [args.data.mu_true[1][1]], s=50, c='g', marker='+')
            animation_save_path = os.path.join(experiment_dir,'animation.gif')
            ani.save(animation_save_path, writer='pillow', fps=30) # Save animation as gif
            logging.info("Animation saved to '{}'.".format(animation_save_path))
            plt.close()


        ## Visualization - trajectory
        if args.saving.save_trajectory:
            logging.info("Saving Point Trajectory...")
            from matplotlib.collections import LineCollection
            import matplotlib.colors as mcolors
            point_trajectory = all_generated_samples[:,0,:] # Shape: (-1, 2)
            point_pairs = point_trajectory.reshape(-1, 1, 2)
            segments = np.concatenate([point_pairs[:-1], point_pairs[1:]], axis=1) # An array where each element is a pair of points

            t = np.linspace(0, 1, point_trajectory.shape[0])
            lc = LineCollection(segments, cmap=plt.get_cmap('viridis'), norm=mcolors.Normalize(0, 1))
            lc.set_array(t)
            lc.set_linewidth(0.5)
            lc.set_label('Trajectory')

            plt.figure(figsize=args.visualization.figsize)
            plt.gca().add_collection(lc)
            plt.scatter([trajectory_start_point[0][0]], [trajectory_start_point[0][1]], s=50, c='r', marker='+', label='Start point')
            plt.legend()
            trajectory_plot_save_path = os.path.join(experiment_dir,'trajectory.svg')
            plt.savefig(trajectory_plot_save_path, format='svg')
            logging.info("Trajectory plot saved to '{}'.".format(trajectory_plot_save_path))
            plt.close()
            
            trajectory_npy_save_path = os.path.join(experiment_dir,'trajectory.npy')
            np.save(trajectory_npy_save_path, all_generated_samples[:,0,:])
            logging.info("Trajectory numpy array saved to '{}'.".format(trajectory_npy_save_path))


        ## Visualization - generation metrics
        if args.saving.save_generation_metric_plot:
            logging.info("Saving Generation Metrics Plot...")
            def add_vertical_lines(ax, x_coords):
                for x in x_coords:
                    ax.axvline(x=x, linestyle='--', color='red', alpha=0.3)

            fig, axs = plt.subplots(2, 2, figsize=args.visualization.figsize)

            metric_plot_x_coords = [i*args.sampling.n_frames_each for i in range(args.model.num_classes+1)]
            axs[0,0].set_title('KL divergence')
            axs[0,0].plot(kl_divergences)
            add_vertical_lines(axs[0,0], metric_plot_x_coords)

            axs[0,1].set_title('Log likelihood')
            axs[0,1].plot(log_likelihoods)
            add_vertical_lines(axs[0,1], metric_plot_x_coords)

            axs[1,0].set_title('MMD2')
            axs[1,0].plot(mmd2s)
            add_vertical_lines(axs[1,0], metric_plot_x_coords)

            axs[1,1].set_title('Wasserstein distance')
            axs[1,1].plot(wasserstein_distances)
            add_vertical_lines(axs[1,1], metric_plot_x_coords)

            metric_plot_save_path = os.path.join(experiment_dir, 'metric_plot.png')
            plt.tight_layout() # Adjust subplot spacing to avoid overlap
            plt.savefig(metric_plot_save_path)
            logging.info(f"Generation metric plot saved to '{metric_plot_save_path}'.")
            plt.close()

        ## Visualization - sampler action record
        if args.saving.save_sampler_record_plot:
            logging.info("Saving Sampler Record Plot...")
            sampler_record_dir = os.path.join(experiment_dir,'sampler_record')
            os.makedirs(sampler_record_dir, exist_ok=True)
            logging.info(f"Sampler record directory created at '{sampler_record_dir}'.")

            plt.figure(figsize=(8,6))
            plt.plot(sampler_record_dict['grad_norms'], label='grad_norms')
            plt.plot(sampler_record_dict['e_int_norms'], label='e_int_norms')
            plt.plot(sampler_record_dict['e_diff_norms'], label='e_diff_norms')
            plt.legend()
            plt.yscale('log', base=2)
            grad_pid_norms_plot_save_path = os.path.join(sampler_record_dir, 'grad_pid_norms_plot.png')
            plt.savefig(grad_pid_norms_plot_save_path, bbox_inches='tight')
            logging.info(f"Grad_pid_norms plot saved to '{grad_pid_norms_plot_save_path}'.")
            plt.close()
    
            plt.figure(figsize=(8,6))
            plt.plot(sampler_record_dict['P_term_norms'], label='P_term_norms')
            plt.plot(sampler_record_dict['I_term_norms'], label='I_term_norms')
            plt.plot(sampler_record_dict['D_term_norms'], label='D_term_norms')
            plt.legend()
            plt.yscale('log', base=2)
            pid_term_norms_plot_1_save_path = os.path.join(sampler_record_dir, 'pid_term_norms_plot1.png')
            plt.savefig(pid_term_norms_plot_1_save_path, bbox_inches='tight')
            logging.info(f"PID_term_norms plot 1 saved to '{pid_term_norms_plot_1_save_path}'.")
            plt.close()

            plt.figure(figsize=(8,6))
            plt.plot(sampler_record_dict['PID_term_norms'], label='PID_term_norms')
            plt.plot(sampler_record_dict['noise_term_norms'], label='noise_term_norms')
            plt.plot(sampler_record_dict['delta_term_norms'], label='delta_term_norms')
            plt.legend()
            plt.yscale('log', base=2)
            pid_term_norms_plot_2_save_path = os.path.join(sampler_record_dir, 'pid_term_norms_plot2.png')
            plt.savefig(pid_term_norms_plot_2_save_path, bbox_inches='tight')
            logging.info(f"PID_term_norms plot 2 saved to '{pid_term_norms_plot_2_save_path}'.")
            plt.close()

            plt.figure(figsize=(8,6))
            plt.plot(sampler_record_dict['snrs'], label='snrs')
            plt.legend()
            snr_plot_save_path = os.path.join(sampler_record_dir,'snr_plot.png')
            plt.savefig(snr_plot_save_path, bbox_inches='tight')
            logging.info(f"SNR plot saved to '{snr_plot_save_path}'.")
            plt.close()

            plt.figure(figsize=(8,6))
            plt.plot(sampler_record_dict['image_norms'], label='image_norms')
            plt.legend()
            image_norms_plot_save_path = os.path.join(sampler_record_dir, 'image_norms_plot.png')
            plt.savefig(image_norms_plot_save_path, bbox_inches='tight')
            logging.info(f"Image norms plot saved to '{image_norms_plot_save_path}'.")
            plt.close()


        # Result saving
        logging.info("Saving Experiment Result...")
        result_dict = {
            'experiment_name': args.saving.experiment_name,
            'comment': args.saving.comment,
            'time_string': time_string,

            # Final Metrics averaged over different sampling process (with different seeds)
            'kl_divergence_final': [kl_divergence_final, kl_divergence_final_std],
            'log_likelihood_final': [log_likelihood_final, log_likelihood_final_std],
            'mmd2_rbf_final': [mmd2_final, mmd2_final_std],
            'wasserstein_distance_final': [wasserstein_distance_final, wasserstein_distance_final_std],

            # Final Metrics of different sampling process (with different seeds)
            'kl_divergence_finals': kl_divergence_finals,
            'log_likelihood_finals': log_likelihood_finals,
            'mmd2_rbf_finals': mmd2_finals,
            'wasserstein_distance_finals': wasserstein_distance_finals,

            # Parameters of the first sampling process
            'weights_preds_final': weights_preds[-1].tolist(),
            'mu_preds_final': mu_preds[-1].tolist(),
            'cov_preds_final': cov_preds[-1].tolist(),

            # Metrics of each frame of the first sampling process
            'kl_divergences': kl_divergences,
            'log_likelihoods': log_likelihoods,
            'mmd2s': mmd2s,
            'wasserstein_distances': wasserstein_distances,

            # Parameters of each frame of the first sampling process
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
    with open(os.path.join('configs', 'point.yml'), 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config_dict['saving']['result_dir'] = 'results111'
    args = dict2namespace(config_dict)
    main(args)
