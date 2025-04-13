"""Metrics for quality of generation"""
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.stats
import os
try:
    import ot
except ImportError:
    print("Error: Please install POT library by running the command 'pip install POT'.")
    exit()
os.environ["OMP_NUM_THREADS"] = "1" # To avoid the warning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.


def sample_wasserstein_distance(X, Y, p=1, numItermax=1000000):
    """
    Wasserstein distance between two groups of samples.
    
    Args:
        X (np.ndarray): Array of shape (n_x, d).
        Y (np.ndarray): Array of shape (n_y, d).
        p (int): Wasserstein distance power. Options: [1, 2].
        numItermax (int): Maximum number of iterations for the OT solver.
    
    Returns:
        out (float): Wasserstein distance between the two distributions.
    """
    a = np.ones(len(X)) / len(X) # Set probability mass (uniform distribution)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y, metric='euclidean') # Euclidean distance matrix of shape (n_x, n_y), where M_{ij} = d(x_i, y_j).

    if p == 1:
        return ot.emd2(a, b, M, numItermax=numItermax)

    if p == 2:
        return np.sqrt(ot.emd2(a, b, M ** 2, numItermax=numItermax))


def sample_mmd2_rbf(X, Y, sigma=1.0):
    """
    Compute the squared maximum mean discrepancy (MMD) between two groups of samples using the RBF kernel.
    
    Reference: https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf, page 6, lemma 6.
    Note that the MMD^2 is not necessarily non-negative, since its is just an unbiased estimator. For details, see page 7 of the reference paper.

    Args:
        X (np.ndarray): Array of shape (nx, d).
        Y (np.ndarray): Array of shape (ny, d).
        sigma (float): Bandwidth of the kernel.
    
    Returns:
        mmd_sq (np.float64): Square MMD.
    """
    def rbf_kernel_distance(x, y, sigma=1.0):
        # x (np.ndarray): array of shape (nx, d)
        # y (np.ndarray): array of shape (ny, d)
        x_sqnorms = np.sum(x**2, axis=1) # shape: (nx,)
        y_sqnorms = np.sum(y**2, axis=1) # shape: (ny,)
        distances = x_sqnorms[:, None] + y_sqnorms[None, :] - 2 * np.dot(x, y.T) # shape: (nx, ny)
        return np.exp(-distances / (2 * sigma**2)) # shape: (nx, ny)
    
    m, n = X.shape[0], Y.shape[0]
    
    # Compute kernel matrices
    K_XX = rbf_kernel_distance(X, X, sigma)
    K_YY = rbf_kernel_distance(Y, Y, sigma)
    K_XY = rbf_kernel_distance(X, Y, sigma)
    
    # (Discard diagonal entries of K_XX and K_YY)
    mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1)) + \
             (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) - \
             2 * np.sum(K_XY) / (m * n)
    return mmd_sq # MMD^2


def gmm_estimation(data, n_components=2):
    """
    Use Gaussian Mixture Model (GMM) to estimate the parameters of a mixture of d-dimensional Gaussians.
    
    Args:
        data (np.ndarray): Array of shape (n_samples, d).
        n_components (int): The number of mixture components (clusters).
    
    Returns:
        out (tuple of np.ndarray): A tuple containing
            - weights_fit (np.ndarray): Array of shape (n_components,).
            - mu_fit (np.ndarray): Array of shape (n_components, d).
            - cov_fit (np.ndarray): Array of shape (n_components, d, d).
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(data)

    weights_fit = gmm.weights_ # proportion of each component in the mixture
    mu_fit = gmm.means_ # shape (n_components, d)
    cov_fit = gmm.covariances_ # shape (n_components, d, d)

    return  weights_fit, mu_fit, cov_fit


def gmm_logpdf(x, weights, means, covs):
    """
    Compute the log-pdf of a set of samples under a d-dimensional GMM with given parameters.

    Args:
        x (np.ndarray): Array of shape (n_samples, d).
        weights (np.ndarray): Array of shape (n_components,).
        means (np.ndarray): Array of shape (n_components, d).
        covs (np.ndarray): Array of shape (n_components, d, d).
    
    Returns:
        out (np.ndarray): Array of shape (n_samples,), the log-pdf of each sample.
    """
    n_components = len(weights)
    log_probs = np.array([
                    np.log(weights[i]) + scipy.stats.multivariate_normal.logpdf(x, mean=means[i], cov=covs[i]) # log pdf for each component: log(w) + log(N(x; mean, cov)).
                        for i in range(n_components)
                        ])  # Shape: (n_components, n_samples)
    # For numerical stability, we use the log-sum-exp trick to avoid overflow.
    return np.logaddexp.reduce(log_probs, axis=0) # = np.log(np.sum(np.exp(log_probs), axis=0))


def gmm_kl(p_weights, p_means, p_covs, q_weights, q_means, q_covs, n_samples=10000):
    """
    Compute the KL divergence between two Gaussian mixtures of dimension d.

    Args:
        p_weights (np.ndarray): Array of shape (n_components,).
        p_means (np.ndarray): Array of shape (n_components, d).
        p_covs (np.ndarray): Array of shape (n_components, d, d).
        q_weights (np.ndarray): Array of shape (n_components,).
        q_means (np.ndarray): Array of shape (n_components, d).
        q_covs (np.ndarray): Array of shape (n_components, d, d).
        n_samples (int): Number of generated samples for Monte Carlo estimation of KL.
    
    Returns:
        out (float): KL divergence between the two GMMs.
    """
    n_components = len(p_weights)

    # Generate samples from p
    gmm = GaussianMixture(n_components=n_components)
    gmm.weights_ = p_weights
    gmm.means_ = p_means
    gmm.covariances_ = p_covs
    samples, _ = gmm.sample(n_samples) # Shape: (n_samples, d)
    log_p = gmm_logpdf(samples, p_weights, p_means, p_covs) # Shape: (n_samples,)
    log_q = gmm_logpdf(samples, q_weights, q_means, q_covs) # Shape: (n_samples,)
    kl = np.mean(log_p - log_q) # KL = E_{x~p}[log(p(x)) - log(q(x))]
    return kl


def gmm_log_likelihood(x, weights, means, covs):
    """
    Compute the **average** log-likelihood of a set of samples under a GMM with given parameters.

    Args:
        x (np.ndarray): Array of shape (n_samples, d).
        weights (np.ndarray): Array of shape (n_components,).
        means (np.ndarray): Array of shape (n_components, d).
        covs (np.ndarray): Array of shape (n_components, d, d).
    
    Returns:
        out (np.float64): log-likelihood of the samples under the GMM.
    """
    return np.mean(gmm_logpdf(x, weights, means, covs))



if __name__ == '__main__':
    n_samples = 10000
    d = 2

    mu_true = np.array([[1, 1],
                        [6, 6]])
    cov_true = np.array([[[1, 0],
                          [0, 1]],
                         [[1, 0],
                          [0, 1]]])
    weights_true = np.array([0.80, 0.20])

    mu_pred = np.array([[1, 1],
                    [6, 7]])
    cov_pred = np.array([[[1, 0],
                      [0, 1]],
                     [[1, 0],
                      [0, 1]]])
    weights_pred = np.array([0.70, 0.30])

    gmm_true = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm_true.weights_ = weights_true
    gmm_true.means_ = mu_true
    gmm_true.covariances_ = cov_true

    gmm_pred = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
    gmm_pred.weights_ = weights_pred
    gmm_pred.means_ = mu_pred
    gmm_pred.covariances_ = cov_pred

    samples_true, _ = gmm_true.sample(n_samples)
    samples_pred, _ = gmm_pred.sample(n_samples)

    #print(sample_wasserstein_distance(samples_true, samples_pred, p=1))
    #mu_pred, cov_pred, weights_pred = gmm_estimation(samples_true, n_components=2)
    #print(mu_pred, cov_pred, weights_pred)
    import time
    
    for _ in range(20):
        t0=time.time()
        kl = gmm_kl(weights_true, mu_true, cov_true, weights_pred, mu_pred, cov_pred, n_samples=100000)
        t1=time.time()
        #print(t1-t0)
        print(kl)
    #print(sample_mmd2_rbf(samples_true, samples_pred))
    #print(log_likelihood(samples_true, weights_true, mu_true, cov_true))

