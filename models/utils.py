import torch
from scipy import stats as sts

from copulae import GaussianCopula


def RMSE(true, pred):
    return torch.sqrt(torch.mean((torch.Tensor(true).reshape(-1) - torch.Tensor(pred).reshape(-1))**2))


def MC_se(x, B):
    return sts.t.ppf(0.975, B - 1) * torch.std(torch.Tensor(x).reshape(-1)) / B


# Train-Test Split
def tr_te_split(X, split):
    train = X[split]
    test = X[~split]

    return train, test


# A Data Generating Process
def example_data(N=1000, P=10):

    # Generate X with a little bit of correlation b/w the continuous variables (i.e. 0.3 Pearson coeffincient c.a.)
    X = torch.zeros((N, P))

    # Generate Gaussian copula correlated uniforms
    g_cop = GaussianCopula(dim=P)

    mycova = torch.ones((P, P))

    for i in range(P):
        for j in range(P):

            mycova[i, j] = 0.1**(torch.abs(torch.scalar_tensor(i - j))) + torch.where(torch.scalar_tensor(i) == torch.scalar_tensor(j), 0.0, 0.1)

    g_cop[:] = mycova

    rv = g_cop.random(N)

    # Generate correlated covariates (2 out of 10)
    X[:, 0:5] = torch.tensor([sts.uniform.ppf(rv[:, i], loc=-3, scale=6) for i in range(5)]).T
    X[:, 5:P] = torch.tensor([sts.binom.ppf(rv[:, i], n=1, p=0.3) for i in range(5, P)]).T

    # Generate A
    und_lin = -1.5 + 0.5*X[:, 0] + sts.uniform.rvs(size=N)/10
    pscore = sts.logistic.cdf(und_lin)

    # sts.describe(pscore)
    # sns.distplot(pscore)

    A = sts.binom.rvs(1, pscore)

    # Generate Y
    mu = 6 + 0.3*torch.exp(X[:, 0]) + 1*X[:, 1]**2 + 1.5*torch.abs(X[:, 2]) + 0.8*X[:, 3]
    ITE = 3 + 0.8*X[:, 0]**2

    sigma = 0.5
    Y = mu + ITE*A + sts.norm.rvs(0, sigma, N)

    return Y, X, torch.Tensor(A), ITE

