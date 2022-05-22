import torch
import math


class BlockLinear(torch.nn.Module):
    def __init__(self, n_blocks, in_features, out_features):
        super().__init__()
        self.n_blocks = n_blocks
        self.in_features = in_features
        self.out_features = out_features
        self.block_weights = []
        self.block_biases = []

        for i in range(n_blocks):

            # Weights
            block_weight = torch.Tensor(out_features, in_features)
            block_weight = torch.nn.Parameter(block_weight)
            torch.nn.init.kaiming_uniform_(block_weight)
            self.register_parameter(
                f'block_weight_{i}',
                block_weight
            )
            self.block_weights.append(block_weight)

            # Bias
            block_bias = torch.Tensor(out_features)
            block_bias = torch.nn.Parameter(block_bias)
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(block_bias, -bound, bound)
            self.register_parameter(
                f'block_bias_{i}',
                block_bias
            )
            self.block_biases.append(block_bias)

    def forward(self, x):
        block_size = x.size(1) // self.n_blocks
        x_blocks = torch.split(
            x,
            split_size_or_sections=block_size,
            dim=1
        )
        block_outputs = []
        for block_id in range(self.n_blocks):
            block_outputs.append(
                x_blocks[block_id] @ self.block_weights[block_id].t() + self.block_biases[block_id]
            )
        return torch.cat(block_outputs, dim=1)


class R_sep_DNN(torch.nn.Module):
    def __init__(self, D_in, H_mu, H_tau, D_out):
        super().__init__()

        self.features_mu = torch.nn.Sequential(
            torch.nn.Linear(D_in, H_mu[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(H_mu[0], H_mu[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(H_mu[1], D_out),
        )

        self.features_tau = torch.nn.Sequential(
            torch.nn.Linear(D_in, H_tau[0]),
            torch.nn.Softplus(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(H_tau[0], D_out)
        )

    def forward(self, x):

        mu = self.features_mu(x)
        tau_Y = self.features_tau(x)

        return mu, tau_Y


class R_DNN(torch.nn.Module):
    def __init__(self, D_in, D_hid, D_trt):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_hid[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(D_hid[0], D_hid[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(D_hid[1], D_trt),
        )

    def forward(self, x):

        unpack = self.features(x)
        return unpack


class R_sep_NAM(torch.nn.Module):
    def __init__(self, n_blocks, H_mu, H_tau, D_out):
        super().__init__()

        self.features_mu = torch.nn.Sequential(
            BlockLinear(n_blocks, 1, H_mu[0]),
            torch.nn.ReLU(),
            BlockLinear(n_blocks, H_mu[0], H_mu[1]),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            BlockLinear(n_blocks, H_mu[1], D_out),
        )

        self.features_tau = torch.nn.Sequential(
            BlockLinear(n_blocks, 1, H_tau[0]),
            torch.nn.Softplus(),
            torch.nn.Dropout(0.2),
            BlockLinear(n_blocks, H_tau[0], D_out)
        )

        self.lr = torch.nn.Linear(n_blocks, D_out)

    def forward(self, x):

        mu = self.features_mu(x)
        tau_Y = self.features_tau(x)

        return self.lr(mu), self.lr(tau_Y)


class R_NAM(torch.nn.Module):
    def __init__(self, n_blocks, D_hid, D_trt):
        super().__init__()

        self.features = torch.nn.Sequential(
            BlockLinear(n_blocks, 1, D_hid[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            BlockLinear(n_blocks, D_hid[0], D_hid[1]),
            torch.nn.ReLU(),
            BlockLinear(n_blocks, D_hid[1], D_trt),
        )

        self.lr = torch.nn.Linear(n_blocks*D_trt, D_trt)

    def forward(self, x):

        unpack = self.features(x)
        return self.lr(unpack)


class R_mix_NAM_DNN(torch.nn.Module):
    def __init__(self, D_in, n_blocks, H_mu, H_tau, D_out):
        super().__init__()

        self.features_mu = torch.nn.Sequential(
            torch.nn.Linear(D_in, H_mu[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(H_mu[0], H_mu[1]),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(H_mu[1], D_out),
        )

        self.features_tau = torch.nn.Sequential(
            BlockLinear(n_blocks, 1, H_tau[0]),
            torch.nn.Softplus(),
            torch.nn.Dropout(0.2),
            BlockLinear(n_blocks, H_tau[0], D_out)
        )

        self.lr = torch.nn.Linear(n_blocks, D_out)

    def forward(self, x):

        mu = self.features_mu(x)
        tau_Y = self.features_tau(x)

        return mu, self.lr(tau_Y)


