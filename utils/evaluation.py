from utils.metrics import sample_wasserstein_distance, gmm_estimation, sample_mmd2_rbf, gmm_kl, gmm_log_likelihood
from datasets.point import generate_point_dataset

frame_indices = np.linspace(0, len(all_generated_samples)-1, args.visualization.n_frames, dtype=int)
frame_samples = all_generated_samples[frame_indices] # Select some samples for evaluation, and for animation frames
logger.info("Frame samples shape: {}".format(frame_samples.shape)) # (n_frames, n_test_samples, 2)
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
    kl_divergences.append(gmm_kl(args.data.weights_true, args.data.mu_true, args.data.cov_true,
                                        weights_preds[i], mu_preds[i], cov_preds[i], n_samples=100000))

log_likelihoods = []
for i, t in tqdm.tqdm(enumerate(frame_indices), desc='Evaluating log likelihood...'):
    generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
    log_likelihoods.append(gmm_log_likelihood(generated_samples, weights_preds[i], mu_preds[i], cov_preds[i]))

mmd2s = []
for t in tqdm.tqdm(frame_indices, desc='Evaluating MMD...'):
    generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
    mmd2s.append(sample_mmd2_rbf(generated_samples, true_samples))

wasserstein_distances = []
for t in tqdm.tqdm(frame_indices, desc='Evaluating Wasserstein distance...'):
    generated_samples = all_generated_samples[t] # (args.data.n_test_samples, 2)
    wasserstein_distances.append(sample_wasserstein_distance(generated_samples, true_samples))


logger.info("Final KL divergence: {}".format(kl_divergences[-1]))
logger.info("Final Log likelihood: {}".format(log_likelihoods[-1]))
logger.info("Final MMD2: {}".format(mmd2s[-1]))
logger.info("Final Wasserstein Distance: {}".format(wasserstein_distances[-1]))
print("Predicted GMM parameters: \n", mu_pred, "\n", cov_pred, "\n", weights_pred)

# Repeat for different seeds
kl_divergence_finals = [kl_divergences[-1]]
log_likelihood_finals = [log_likelihoods[-1]]
mmd2_finals = [mmd2s[-1]]
wasserstein_distance_finals = [wasserstein_distances[-1]]

for seed in [42, 123, 456]:
    set_seed(seed)
    initial_noise = (16*torch.rand(args.data.n_test_samples,2,generator=gen)-8).to('cpu') # uniformly sample from [-8, 8]
    reconstructed_samples = sampler(initial_noise.to(args.device), score, sigmas,
                                                    n_steps_each=args.sampling.n_steps_each,
                                                    step_lr=args.sampling.step_lr,
                                                    verbose=False,
                                                    final_only=True)[0].cpu().detach().numpy()
    logger.info("Reconstructed samples shape: {}".format(reconstructed_samples.shape)) # (n_test_samples, 2)

    mu_pred, cov_pred, weights_pred = gmm_estimation(reconstructed_samples)
    kl_divergence = gmm_kl(args.data.weights_true, args.data.mu_true, args.data.cov_true, weights_pred, mu_pred, cov_pred, n_samples=100000)
    log_likelihood = gmm_log_likelihood(reconstructed_samples, weights_pred, mu_pred, cov_pred)
    mmd2 = sample_mmd2_rbf(reconstructed_samples, true_samples)
    wasserstein_distance = sample_wasserstein_distance(reconstructed_samples, true_samples)

    kl_divergence_finals.append(kl_divergence)
    log_likelihood_finals.append(log_likelihood)
    mmd2_finals.append(mmd2)
    wasserstein_distance_finals.append(wasserstein_distance)

    logger.info("Seed {}: KL divergence: {}".format(seed, kl_divergence))
    logger.info("Seed {}: Log likelihood: {}".format(seed, log_likelihood))
    logger.info("Seed {}: MMD2: {}".format(seed, mmd2))
    logger.info("Seed {}: Wasserstein Distance: {}".format(seed, wasserstein_distance))








