import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.stats

try:
    import ot
except ImportError:
    print("Please install POT library by running the command 'pip install POT'.")
    exit()


def sample_wasserstein_distance(X, Y, p=1, numItermax=1000000):
    """
    Wasserstein distance between two groupd of samples.
    Args:
        X (np.ndarray): array of shape (n_x, d)
        Y (np.ndarray): array of shape (n_y, d)
        p (int): Wasserstein distance power. Options: [1, 2].
        numItermax (int): Maximum number of iterations for the OT solver.
    """
    a = np.ones(len(X)) / len(X) # Set probability mass (uniform distribution)
    b = np.ones(len(Y)) / len(Y)
    M = ot.dist(X, Y, metric='euclidean') # Euclidean distance matrix of shape (n_x, n_y),, where M_{ij} = d(x_i, y_j).

    if p == 1:
        return ot.emd2(a, b, M, numItermax=numItermax)

    if p == 2:
        return np.sqrt(ot.emd2(a, b, M ** 2))


def gmm_estimation(data, n_components=2):
    """
    Use Gaussian Mixture Model (GMM) to estimate the parameters of a mixture of Gaussians.
    
    Args:
        data (np.ndarray): array of shape (n_samples, n_features).
        n_components (int): The number of mixture components (clusters).
    
    Returns:
        out (tuple of np.ndarray): A tuple containing:
        - mu_fit (np.ndarray): array of shape (n_components, n_features)
        - cov_fit (np.ndarray): array of shape (n_components, n_features, n_features)
        - weights_fit (np.ndarray): array of shape (n_components,)
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.fit(data)

    mu_fit = gmm.means_ # shape (n_components, n_features)
    cov_fit = gmm.covariances_ # shape (n_components, n_features, n_features)
    weights_fit = gmm.weights_ # proportion of each component in the mixture

    return mu_fit, cov_fit, weights_fit



def rbf_kernel(x, y, sigma=1.0):
    """
    Compute the RBF kernel between two matrices x and y.

    Args:
        x (np.ndarray): array of shape (nx, d)
        y (np.ndarray): array of shape (ny, d)
        sigma (float): bandwidth of the kernel.

    Returns:
        np.ndarray: kernel matrix of shape (nx, ny)
    """
    x_sqnorms = np.sum(x**2, axis=1) # shape: (nx,)
    y_sqnorms = np.sum(y**2, axis=1) # shape: (ny,)
    distances = x_sqnorms[:, None] + y_sqnorms[None, :] - 2 * np.dot(x, y.T) # shape: (nx, ny)
    return np.exp(-distances / (2 * sigma**2))


def mmd_rbf(X, Y, sigma=1.0):
    """
    Compute the maximum mean discrepancy (MMD) between two groups of samples using the RBF kernel.

    Args:
        X (np.ndarray): array of shape (n_x, d)
        Y (np.ndarray): array of shape (n_y, d)
        sigma (float): bandwidth of the kernel.
    """
    m, n = X.shape[0], Y.shape[0]
    
    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    
    # (Discard diagonal entries of K_XX and K_YY)
    mmd_sq = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1)) + \
             (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) - \
             2 * np.sum(K_XY) / (m * n)
    return mmd_sq # MMD^2


def kl_2d_gmms(p_weights, p_means, p_covs, q_weights, q_means, q_covs, n_samples=10000):
    """
    Compute the KL divergence between two 2D Gaussian mixtures.

    Args:
        p_weights (np.ndarray): array of shape (n_components,)
        p_means (np.ndarray): array of shape (n_components, 2)
        p_covs (np.ndarray): array of shape (n_components, 2, 2)
        q_weights (np.ndarray): array of shape (n_components,)
        q_means (np.ndarray): array of shape (n_components, 2)
        q_covs (np.ndarray): array of shape (n_components, 2, 2)
        n_samples (int): number of samples to generate.
    
    Returns:
        out (float): KL divergence between the two GMMs.
    """
    n_components = len(p_weights)

    # Generate samples from p1
    samples = []
    for _ in range(n_samples):
        component = np.random.choice(n_components, p=p_weights) # int
        mean = p_means[component] # Choose a mean from a component of GMM 1
        cov = p_covs[component] # Choose a covariance from a component of GMM 1
        sample = np.random.multivariate_normal(mean, cov)
        samples.append(sample)
    samples = np.array(samples) # Shape: (n_samples, 2)

    log_p = []
    for w, mean, cov in zip(p_weights, p_means, p_covs):
        log_p.append(np.log(w) + scipy.stats.multivariate_normal.logpdf(samples, mean, cov)) # Append shape: (n_samples,)
    log_p = np.logaddexp.reduce(log_p, axis=0) # Shape: (n_samples,)

    log_q = []
    for w, mean, cov in zip(q_weights, q_means, q_covs):
        log_q.append(np.log(w) + scipy.stats.multivariate_normal.logpdf(samples, mean, cov)) # Append shape: (n_samples,)
    log_q = np.logaddexp.reduce(log_q, axis=0) # Shape: (n_samples,)

    kl = np.mean(log_p - log_q)
    return kl


if __name__ == '__main__':
    np.random.seed(11110)
    n = 10000
    d = 2

    mu_true = np.array([[1, 1],
                        [6, 6]])
    cov_true = np.array([[[1, 0],
                          [0, 1]],
                         [[1, 0],
                          [0, 1]]])
    weights_true = np.array([0.80, 0.20])

    samples_true_1 = np.random.multivariate_normal(mu_true[0], cov_true[0], size=(int(n*weights_true[0]),))
    samples_true_2 = np.random.multivariate_normal(mu_true[1], cov_true[1], size=(int(n*weights_true[1]),))
    data = np.vstack([samples_true_1, samples_true_2])

    #print(sample_wasserstein_distance(samples_1, samples_2, p=1))
    mu_pred, cov_pred, weights_pred = gmm_estimation(data, n_components=2)
    kl = kl_2d_gmms(weights_true, mu_true, cov_true, weights_pred, mu_pred, cov_pred)
    print(kl)
    #print(mmd_rbf(samples_1, samples_2))


