import os
from models.utils import *
from models.NAM import *
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

# Deprecated but necessary here
import warnings
warnings.filterwarnings("ignore")

# Load data
print(os.getcwd())
ACTG_df = pd.read_csv('./Data/ACTG175.csv')

assert not np.any(pd.isna(ACTG_df))

Z = np.array(ACTG_df['treat'])
Y = np.array(ACTG_df['Y'])
X = np.array(ACTG_df.drop(columns=['Y', 'treat']))

col_names = list(ACTG_df.drop(columns=['Y', 'treat']).columns)

# Standardize Y(= difference in cell counts after 20+/-5 weeks from baseline) and continuous X
scaler = StandardScaler()
# Y_scale = scaler.fit_transform(Y.reshape(-1, 1))
X[:, (0, 1, 7)] = scaler.fit_transform(X[:, (0, 1, 7)])

N, P = X.shape

# ICNN model
mod_ICNN = R_sep_NAM(n_blocks=P, H_mu=[5, 5], H_tau=[2], D_out=1)
optimizer_ICNN = torch.optim.Adam(mod_ICNN.parameters(), lr=0.5)
criterion_ICNN = torch.nn.MSELoss()

###### R-sep-NZM
mod_ICNN.train()
for i in range(2000):
    optimizer_ICNN.zero_grad()
    mu, tau_Z = mod_ICNN(torch.Tensor(X))
    y_pred = mu + tau_Z * torch.Tensor(Z).reshape(-1, 1)
    loss = criterion_ICNN(torch.Tensor(Y).float(), y_pred.reshape(-1, 1))
    loss.backward()
    optimizer_ICNN.step()
    if i % 100 == 0:
        print(loss)

# Pred out of sample
with torch.no_grad():

    mod_ICNN.eval()
    _, pred_ITE_A_ICNN = mod_ICNN(torch.Tensor(X))


# Plot Mediating Effects
# R-sep-NAM visualizer on tau_A and tau_M for interpretability
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 10})

x = torch.linspace(-3, 3, 5000).reshape(-1, 1)
x = torch.hstack(P*[x])

# Replace binary variables
x[:, np.r_[2:7, 8:12]] = torch.Tensor(np.c_[9*[np.r_[np.zeros(2500), np.ones(2500)]]].transpose())

mod_ICNN.train()

# Tau-A
fig, axs = plt.subplots(2, 6)
for i, name in zip(range(P), col_names):

    special_y = []
    for j in range(100):
        special_y.append(mod_ICNN.get_submodule('lr').weight[0][i].item() *
                         mod_ICNN.get_submodule('features_tau')(x)[:, i].detach().numpy())

    special_y = np.array(special_y).transpose()

    if i < 6:
        axs[0, i].plot(x[:, 0].detach().numpy(),
                       np.mean(special_y, axis=1), label='Estimated', c='r')
        axs[0, i].fill_between(x[:, 0].detach().numpy(), np.mean(special_y, axis=1) - 1.96*np.std(special_y, axis=1),
                                   np.mean(special_y, axis=1) + 1.96*np.std(special_y, axis=1),
                                   color='grey', alpha=0.4, label='MC bands')
        axs[0, i].set_title(name)
        # axs[0, i].legend(bbox_to_anchor=(0.45, 1.01))

        if i in np.r_[2:7]:
            axs[0, i].set_xticks([-2.5, 0, 2.5])
            axs[0, i].set_xticklabels(["0", "", "1"])


    elif i >= 6:
        axs[1, 11-i].plot(x[:, 0].detach().numpy(),
                          np.mean(special_y, axis=1), label='Estimated', c='r')
        axs[1, 11-i].fill_between(x[:, 0].detach().numpy(), np.mean(special_y, axis=1) - 1.96 * np.std(special_y, axis=1),
                                   np.mean(special_y, axis=1) + 1.96 * np.std(special_y, axis=1),
                                   color='grey', alpha=0.4, label='MC bands')
        axs[1, 11-i].set_title(name)
        # axs[1, i].legend(bbox_to_anchor=(0.45, 1.01))

        if i in np.r_[8:12]:
            axs[1, 11-i].set_xticks([-2.5, 0, 2.5])
            axs[1, 11-i].set_xticklabels(["0", "", "1"])

axs[0, 0].set(ylabel='CATE')
axs[1, 0].set(ylabel='CATE')

axs[0, 1].set_yticks(np.linspace(-0.4940, -0.4900, num=6))

for ax in axs.flat:
    ax.set(xlabel='X')

plt.subplots_adjust(left=0.04,
                    bottom=0.21,
                    right=0.995,
                    top=0.81,
                    wspace=0.31,
                    hspace=0.4)

plt.savefig('./Data/CATE_cova.pdf')



