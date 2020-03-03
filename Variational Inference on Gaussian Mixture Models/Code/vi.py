# author Olga Mikheeva olgamik@kth.se
# PGM tutorial on Variational Inference
# Bayesian Mixture of Gaussians

import numpy as np
import matplotlib.pyplot as plt
import math
import random

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
    plt.savefig('Plots/elbo'+title)
    plt.show()


def plot_elbo(elbo):
    plt.plot(elbo)
    plt.title('ELBO')
    plt.savefig('Plots/elbo')
    plt.show()


def compute_elbo(data, psi, m, s2, sigma2, mu0):
    """ Computes ELBO """
    n, p = data.shape
    k = m.shape[0]
    m2=np.zeros(k)
    for _ in range(len(m2)):
        m2[_]=np.dot(m[_],m[_].T)
    t1 = np.log(s2) - m2/sigma2
    t1 = t1.sum()
    t2 = -0.5*(m2+s2.T)
    t2 = np.dot(data, m.T) +t2
    t2 -= np.log(psi)
    t2 *= psi
    t2 = t2.sum()
    elbo=t1+t2
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
    # np.random.seed(123)
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
        term1=np.dot(data,m.T) # dim is iXk so 500x5
        m2=np.zeros(k)
        for _ in range(len(m2)):
            m2[_]=np.dot(m[_],m[_].T)
        term2=-0.5*(m2+s2.T)
        psi=np.exp(term1+term2)
        psi_sums = np.sum(psi,axis=1)
        psi = psi / psi_sums[:, np.newaxis]
        # TODO: update posterior parameters for the component means
        m_num=np.matmul(psi.T,data)
        m_den=((1/sigma2)+np.sum(psi,axis=0).reshape(-1,1))
        m=m_num/m_den
        s2=1/m_den
        # compute ELBO
        elbo.append(compute_elbo(data, psi, m, s2, sigma2, m0))
        convergence = elbo[-1] - elbo[-2]

    return m, s2, psi, elbo


def main():
    # parameters
    p = 2
    k = 7
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
        # t=str(i)
        # class_pred = np.argmax(psi_i, axis=1)
        # plot(data[:, 0], data[:, 1],class_pred ,m_i, title=t)
        # plt.show()
        if i > 0 and elbo[-1][-1] > elbo[best_i][-1]:
            best_i = i
    class_pred = np.argmax(psi[best_i], axis=1)
    plot(data[:, 0], data[:, 1], categories, means, title='true data ')
    plot(data[:, 0], data[:, 1], class_pred, m[best_i], title='posterior ')
    plot_elbo(elbo[best_i])

if __name__ == '__main__':
    main()
