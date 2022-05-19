import os
from models.utils import *
from models.NAM import *
from copy import deepcopy
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression

from econml.dml import CausalForestDML

# Deprecated but necessary here
import warnings
warnings.filterwarnings("ignore")

RMSE_mod_Train = {'S-DNN': [], 'T-DNN': [], 'Causal Forest': [], 'R-DNN': [],
                  'R-sep-DNN': [], 'R-NAM': [], 'R-sep-NAM': [], 'R-sep-mix': []}
RMSE_mod_Test = {'S-DNN': [], 'T-DNN': [], 'Causal Forest': [], 'R-DNN': [],
                 'R-sep-DNN': [], 'R-NAM': [], 'R-sep-NAM': [], 'R-sep-mix': []}

N, P, B = 2000, 10, 100

for b in range(B):

    torch.manual_seed(b*50+10)
    torch.cuda.manual_seed(b*50+10)
    np.random.seed(b*50+10)

    print('\n\n Iteration %s' % (b+1))

    # Generate Data
    Y, X, A, ITE = example_data(N=N, P=P)
    Y = Y.reshape(-1, 1)

    # Train-Test Split (70-30%)
    split = torch.multinomial(torch.Tensor([0.3, 0.7]), N, replacement=True).bool()

    X_train, X_test = tr_te_split(X, split)
    A_train, A_test = tr_te_split(A, split)
    Y_train, Y_test = tr_te_split(Y, split)

    ITE_train, ITE_test = tr_te_split(ITE, split)

    # Models unpacking
    mod_RDNN = R_DNN(D_in=X.shape[1], D_hid=[50, 50], D_trt=2)
    mod_sep_RDNN = R_sep_DNN(D_in=X.shape[1], H_mu=[50, 50], H_tau=[20], D_out=1)

    mod_RNAM = R_NAM(n_blocks=P, D_hid=[20, 20], D_trt=2)
    mod_sep_RNAM = R_sep_NAM(n_blocks=P, H_mu=[20, 20], H_tau=[20], D_out=1)

    mod_Rmix = R_mix_NAM_DNN(D_in=X.shape[1], n_blocks=P, H_mu=[50, 50], H_tau=[20], D_out=1)

    # Define optimizers
    optimizer_RDNN = torch.optim.Adam(mod_RDNN.parameters(), lr=0.01)
    optimizer_sep_RDNN = torch.optim.Adam(mod_sep_RDNN.parameters(), lr=0.01)

    optimizer_RNAM = torch.optim.Adam(mod_RNAM.parameters(), lr=0.01)
    optimizer_sep_RNAM = torch.optim.Adam(mod_sep_RNAM.parameters(), lr=0.01)

    optimizer_Rmix = torch.optim.Adam(mod_sep_RNAM.parameters(), lr=0.01)

    # Criterion
    criterion_RDNN = torch.nn.MSELoss()
    criterion_sep_RDNN = torch.nn.MSELoss()

    criterion_RNAM = torch.nn.MSELoss()
    criterion_sep_RNAM = torch.nn.MSELoss()

    criterion_Rmix = torch.nn.MSELoss()


    ###### S-NN
    size = A.shape[0]
    mod_S = MLPRegressor(hidden_layer_sizes=(50, 50), solver='adam', max_iter=2000)
    mod_S.fit(X=np.c_[X_train, A_train], y=Y_train)

    pred_1 = mod_S.predict(X=np.c_[X, np.ones(size)])
    pred_0 = mod_S.predict(X=np.c_[X, np.zeros(size)])

    RMSE_mod_Train['S-DNN'].append(RMSE(ITE_train, (pred_1 - pred_0)[split]))
    RMSE_mod_Test['S-DNN'].append(RMSE(ITE_test, (pred_1 - pred_0)[~split]))


    ###### T-NN
    mod_T = MLPRegressor(hidden_layer_sizes=(50, 50), solver='adam', max_iter=2000)
    mod_dict = {}

    for i in range(2):
            mod_dict['mod_T%s' % i] = deepcopy(mod_T)

    mod_dict['mod_T1'].fit(X=X_train[A_train == 1, :], y=Y_train.reshape(-1)[A_train == 1])
    mod_dict['mod_T0'].fit(X=X_train[A_train == 0, :], y=Y_train.reshape(-1)[A_train == 0])

    pred_1 = mod_dict['mod_T1'].predict(X=X)
    pred_0 = mod_dict['mod_T0'].predict(X=X)

    RMSE_mod_Train['T-DNN'].append(RMSE(ITE_train, (pred_1 - pred_0)[split]))
    RMSE_mod_Test['T-DNN'].append(RMSE(ITE_test, (pred_1 - pred_0)[~split]))


    ###### R-Causal Forest
    causal_forest = CausalForestDML()
    causal_forest.fit(Y=np.array(Y_train).astype(float), T=np.array(A_train).astype(float),
                      X=np.array(X_train).astype(float), W=None)
    pred_CF = causal_forest.effect(X=X)

    RMSE_mod_Train['Causal Forest'].append(RMSE(ITE_train, pred_CF[split]))
    RMSE_mod_Test['Causal Forest'].append(RMSE(ITE_test, pred_CF[~split]))


    ###### R_DNN
    mod_RDNN.train()
    for i in range(2000):
        optimizer_RDNN.zero_grad()
        pred = mod_RDNN(X_train)
        y_pred = pred[:, 0] + pred[:, 1] * A_train
        loss = criterion_RDNN(Y_train.float(), y_pred.reshape(-1, 1))
        loss.backward()
        optimizer_RDNN.step()
        # if i % 500 == 0:
        #     print(loss)


    # Pred out of sample
    with torch.no_grad():

        mod_RDNN.eval()
        pred_ITE_A_RDNN = mod_RDNN(X)[:, 1].reshape(-1)

    RMSE_mod_Train['R-DNN'].append(RMSE(ITE_train, pred_ITE_A_RDNN[split]))
    RMSE_mod_Test['R-DNN'].append(RMSE(ITE_test, pred_ITE_A_RDNN[~split]))


    ###### R-sep-DNN
    mod_sep_RDNN.train()
    for i in range(2000):
        optimizer_sep_RDNN.zero_grad()
        mu, tau_A = mod_sep_RDNN(X_train)
        y_pred = mu + tau_A * A_train.reshape(-1, 1)
        loss = criterion_sep_RDNN(Y_train.float(), y_pred.reshape(-1, 1))
        loss.backward()
        optimizer_sep_RDNN.step()
        # if i % 500 == 0:
        #     print(loss)


    # Pred out of sample
    with torch.no_grad():

        mod_sep_RDNN.eval()
        _, pred_ITE_A_sep_RDNN = mod_sep_RDNN(X)

    RMSE_mod_Train['R-sep-DNN'].append(RMSE(ITE_train, pred_ITE_A_sep_RDNN[split]))
    RMSE_mod_Test['R-sep-DNN'].append(RMSE(ITE_test, pred_ITE_A_sep_RDNN[~split]))



    ###### R_NAM
    mod_RNAM.train()
    for i in range(2000):
        optimizer_RNAM.zero_grad()
        pred = mod_RNAM(X_train)
        y_pred = pred[:, 0] + pred[:, 1] * A_train
        loss = criterion_RNAM(Y_train.float(), y_pred.reshape(-1, 1))
        loss.backward()
        optimizer_RNAM.step()
        # if i % 500 == 0:
        #     print(loss)

    # Pred out of sample
    with torch.no_grad():

        mod_RNAM.eval()
        pred_ITE_A_RNAM = mod_RNAM(X)[:, 1].reshape(-1)

    RMSE_mod_Train['R-NAM'].append(RMSE(ITE_train, pred_ITE_A_sep_RDNN[split]))
    RMSE_mod_Test['R-NAM'].append(RMSE(ITE_test, pred_ITE_A_sep_RDNN[~split]))


    ###### R-sep-NAM
    mod_sep_RNAM.train()
    for i in range(2000):
        optimizer_sep_RNAM.zero_grad()
        mu, tau_A = mod_sep_RNAM(X_train)
        y_pred = mu + tau_A * A_train.reshape(-1, 1)
        loss = criterion_sep_RNAM(Y_train.float(), y_pred.reshape(-1, 1))
        loss.backward()
        optimizer_sep_RNAM.step()
        # if i % 500 == 0:
        #     print(loss)


    # Pred out of sample
    with torch.no_grad():

        mod_sep_RNAM.eval()
        _, pred_ITE_A_sep_RNAM = mod_sep_RNAM(X)

    RMSE_mod_Train['R-sep-NAM'].append(RMSE(ITE_train, pred_ITE_A_sep_RNAM[split]))
    RMSE_mod_Test['R-sep-NAM'].append(RMSE(ITE_test, pred_ITE_A_sep_RNAM[~split]))


    ###### R-sep-mix
    mod_Rmix.train()
    for i in range(2000):
        optimizer_Rmix.zero_grad()
        mu, tau_A = mod_Rmix(X_train)
        y_pred = mu + tau_A * A_train.reshape(-1, 1)
        loss = criterion_Rmix(Y_train.float(), y_pred.reshape(-1, 1))
        loss.backward()
        optimizer_Rmix.step()
        # if i % 500 == 0:
        #     print(loss)


    # Pred out of sample
    with torch.no_grad():

        mod_Rmix.eval()
        _, pred_ITE_A_Rmix = mod_Rmix(X)

    RMSE_mod_Train['R-sep-mix'].append(RMSE(ITE_train, pred_ITE_A_Rmix[split]))
    RMSE_mod_Test['R-sep-mix'].append(RMSE(ITE_test, pred_ITE_A_Rmix[~split]))




print('\n\nTrain CATE:\n')
for i, j in RMSE_mod_Train.items():
    print('%s - RMSE: %s, SE: %s' % (i, np.round(torch.mean(torch.Tensor(j)).item(), 4),
                                     np.round(MC_se(torch.Tensor(j), B).item(), 4)))

print('\n\nTest CATE:\n')
for i, j in RMSE_mod_Test.items():
    print('%s - RMSE: %s, SE: %s' % (i, np.round(torch.mean(torch.Tensor(j)).item(), 4),
                                     np.round(MC_se(torch.Tensor(j), B).item(), 4)))


with open('./Results.txt', 'w') as f:
    f.write('Train CATE:\n')
    for i, j in RMSE_mod_Train.items():
        f.write('\n%s - RMSE: %s, SE: %s' % (i, np.round(torch.mean(torch.Tensor(j)).item(), 4),
                                             np.round(MC_se(torch.Tensor(j), B).item(), 4)))

    f.write('\n\nTest CATE:\n')
    for i, j in RMSE_mod_Test.items():
        f.write('\n%s - RMSE: %s, SE: %s' % (i, np.round(torch.mean(torch.Tensor(j)).item(), 4),
                                             np.round(MC_se(torch.Tensor(j), B).item(), 4)))


# R-sep-NAM visualizer on tau_A and tau_M for interpretability
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 13})

x = torch.linspace(-3, 3, 5000).reshape(-1, 1)
x = torch.hstack(10*[x])

mod_sep_RNAM.train()

# Tau-A
for i in range(1):

    special_y = []
    for j in range(500):
        special_y.append(mod_sep_RNAM.get_submodule('lr').weight[0][i].item() *
                         mod_sep_RNAM.get_submodule('features_tau')(x)[:, i].detach().numpy())


    special_y = np.array(special_y).transpose()

    plt.plot(
        x[:, 0].detach().numpy(),
        np.mean(special_y, axis=1), label='Estimated')
    plt.plot(x[:, 0], 0.8*x[:, 0]**2, '--', c='b', label='True')
    plt.fill_between(x[:, 0].detach().numpy(), np.mean(special_y, axis=1) - 2*np.std(special_y, axis=1),
                     np.mean(special_y, axis=1) + 2*np.std(special_y, axis=1),
                     color='grey', alpha=0.4, label='MC bands')
    plt.title(rf'CATE Covariate $X_{i+1}$')
    plt.legend(bbox_to_anchor=(0.45, 1.01))
    plt.show()

plt.savefig('./Covariate1.pdf')

