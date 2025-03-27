import torch.nn as nn


class SimpleNet1d(nn.Module):
    def __init__(self, data_dim, hidden_dim, sigmas, act=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(data_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, data_dim)
        self.act = act
        self.sigmas = sigmas


    def forward(self, x, y):
        # x: (batch_size, data_dim)
        # y: (batch_size,)
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
        x = self.act(x)
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
        # y: (batch_size,)
        x = self.fc_in(x)
        x = self.bn_in(x)
        x = self.act(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(x)
        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        x = x / used_sigmas
        return x
