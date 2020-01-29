# author Olga Mikheeva olgamik@kth.se
# PGM tutorial on Variational Inference
# Bayesian Mixture of Gaussians

import numpy as np
import matplotlib.pyplot as plt
import math


def generate_data(std, k, n, dim=1):
    means = np.random.normal(0.0, std, size=(k, dim))
    data = []
    categories = []
    for i in range(n):
        cat = np.random.choice(k)  # sample component assignment
        categories.append(cat)
        data.append(np.random.multivariate_normal(means[cat, :], np.eye(dim)))  # sample data point from the Gaussian
    return np.stack(data), categories, means


def plot(x, y, c, means, title):
    plt.scatter(x, y, c=c)
    plt.scatter(means[:, 0], means[:, 1], c='r')
    plt.title(title)
    plt.show()


def plot_elbo(elbo):
    plt.plot(elbo)
    plt.title('ELBO')
    plt.show()


def compute_elbo(data, psi, m, s2, sigma2, mu0):
    """ Computes ELBO """
    n, p = data.shape
    k = m.shape[0]

    elbo = 0

    # TODO: compute ELBO
    # expected log prior over mixture assignments

    # expected log prior over mixture locations

    # expected log likelihood

    # entropy of variational location posterior

    # entropy of the variational assignment posterior

    return elbo


def cavi(data, k, sigma2, m0, eps=1e-15):
    """ Coordinate ascent Variational Inference for Bayesian Mixture of Gaussians
    :param data: data
    :param k: number of components
    :param sigma2: prior variance
    :param m0: prior mean
    :param eps: stopping condition
    :return (m_k, s2_k, psi_i)
    """
    n, p = data.shape
    # initialize randomly
    m = np.random.normal(0., 1., size=(k, p))
    s2 = np.square(np.random.normal(0., 1., size=(k, 1)))
    psi = np.random.dirichlet(np.ones(k), size=n)

    # compute ELBO
    elbo = [compute_elbo(data, psi, m, s2, sigma2, m0)]
    convergence = 1.
    while convergence > eps:  # while ELBO not converged
        # TODO: update categorical

        # TODO: update posterior parameters for the component means

        # compute ELBO
        elbo.append(compute_elbo(data, psi, m, s2, sigma2, m0))
        convergence = elbo[-1] - elbo[-2]

    return m, s2, psi, elbo


def main():
    # parameters
    p = 2
    k = 5
    sigma = 5.

    data, categories, means = generate_data(std=sigma, k=k, n=500, dim=p)
    m = list()
    s2 = list()
    psi = list()
    elbo = list()
    best_i = 0
    for i in range(10):
        m_i, s2_i, psi_i, elbo_i = cavi(data, k=k, sigma2=sigma, m0=np.zeros(p))
        m.append(m_i)
        s2.append(s2_i)
        psi.append(psi_i)
        elbo.append(elbo_i)
        if i > 0 and elbo[-1][-1] > elbo[best_i][-1]:
            best_i = i
    class_pred = np.argmax(psi[best_i], axis=1)
    plot(data[:, 0], data[:, 1], categories, means, title='true data')
    plot(data[:, 0], data[:, 1], class_pred, m[best_i], title='posterior')
    plot_elbo(elbo[best_i])

if __name__ == '__main__':
    main()
