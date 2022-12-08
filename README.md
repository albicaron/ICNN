# `ICNN`

[PyTorch](https://pytorch.org/) implementation of Interpretable Causal Neural Networks (ICNN) as in [Caron et al. (2022)](https://arxiv.org/pdf/2206.10261.pdf). Code replicates all example and simulation studies in the paper.

## Interpretability in Estimating Causal Effects

ICNN is a very flexible deep learning architecture to estimate heterogeneous treatment/causal effects. It combines flexibility in approximating complex non-linear functions through neural nets, and interpretability aimed at returning measure of feature importance as to what are the main moderators of causal effects across units, i.e. what are the main factors driving heterogeneity in the response to a treatment/intervention. ICNN is based on Neural Additive Models [(Agarwal et al., 2021)](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf).
