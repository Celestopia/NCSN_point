import torch
import torch.nn as nn
import numpy as np


class SimpleNet1d(nn.Module):
    """A simple 3-layer MLP"""
    # Reference: https://github.com/ermongroup/ncsn/blob/master/runners/toy_runner.py, line 198.
    def __init__(self, data_dim, hidden_dim, sigmas, act=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)
        self.act = act
        self.sigmas = sigmas


    def forward(self, x, y):
        # x: (batch_size, data_dim)
        # y: (batch_size,), noise level index for each sample
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        x = x / used_sigmas
        return x


class SimpleResidualBlock(nn.Module):
    def __init__(self, data_dim, act=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, data_dim)
        self.bn1 = nn.BatchNorm1d(data_dim)
        self.fc2 = nn.Linear(data_dim, data_dim)
        self.bn2 = nn.BatchNorm1d(data_dim)
        self.act = act
    
    def forward(self, x):
        res = x
        res = self.fc1(res)
        res = self.bn1(res)
        res = self.act(res)
        res = self.fc2(res)
        res = self.bn2(res)
        x = x + res
        #x = self.act(x)
        return x

class SimpleResNet(nn.Module):
    def __init__(self, data_dim, hidden_dim, sigmas, act=nn.ReLU(), num_blocks=3):
        super().__init__()
        self.fc_in = nn.Linear(data_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([SimpleResidualBlock(hidden_dim, act=act) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, data_dim)
        self.act = act
        self.sigmas = sigmas
    
    def forward(self, x, y):
        # x: (batch_size, data_dim)
        # y: (batch_size,), noise level index for each sample
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        x = x / used_sigmas
        return x



class GMMDist:
    """Ground Truth GMM distribution score model."""
    def __init__(self,
                weights=[0.8, 0.2],
                means=[[5.0, 5.0], [-5.0, -5.0]],
                covs=[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
                ):
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.means = torch.tensor(means, dtype=torch.float32)
        self.covs = torch.tensor(covs, dtype=torch.float32)

    def sample(self, n):
        """Sample from the GMM distribution. Output Shape: (n, 2)."""
        mix_indices = np.random.choice(len(self.weights), size=n, p=self.weights.numpy())
        points = []
        for i in mix_indices:
            mean = self.means[i].numpy()
            cov = self.covs[i].numpy()
            L = np.linalg.cholesky(cov)
            z = np.random.randn(2)
            point = mean + L @ z
            points.append(point)
        return torch.tensor(np.array(points), dtype=torch.float32)

    def log_prob(self, samples):
        """Log probability of the GMM distribution. Output Shape: (samples.shape[0],)."""
        # samples: (B, 2)
        logps = []
        d = 2
        for i in range(len(self.weights)):
            # Compute component parameters
            mean = self.means[i]
            cov = self.covs[i]
            cov_inv = torch.inverse(cov)
            log_det = torch.logdet(cov)
            diff = samples - mean
            mahalanobis = (diff @ cov_inv * diff).sum(dim=-1) # (B, 2) @ (2, 2) * (B, 2) -> (B, 2) * (B, 2) -> (B, 2) -sum-> (B,)
            log_p = -0.5 * (mahalanobis + log_det + d * torch.log(torch.tensor(2 * np.pi))) # Log probability for this component
            log_p_weighted = log_p + torch.log(self.weights[i]) # (B,)
            logps.append(log_p_weighted)
        
        # Combine components
        logps = torch.stack(logps) # (2, B)
        return torch.logsumexp(logps, dim=0) # = torch.log(torch.sum(torch.exp(log_probs), axis=0))

    def score(self, samples):
        """Score function of the GMM distribution. Output Shape: (samples.shape[0], 2)."""
        samples = samples.detach().requires_grad_(True)
        log_probs = self.log_prob(samples).sum()
        return torch.autograd.grad(log_probs, samples)[0]

    def forward(self, x, y):
        return self.score(x)

    def __call__(self, x, y):
        return self.forward(x, y)




if __name__ == '__main__':
    gmm = GMMDist()
    samples = gmm.sample(1)
    log_probs = gmm.log_prob(samples)
    scores = gmm.score(samples)
    print(log_probs)
    print(scores)

