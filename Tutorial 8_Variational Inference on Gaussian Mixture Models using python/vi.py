# -*- coding: UTF-8 -*-

"""
Coordinate Ascent Variational Inference process to
approximate a Mixture of Gaussians (GMM) with known precisions
"""

from __future__ import absolute_import

import argparse
import math
import os
import pickle as pkl
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import dirichlet_expectation, log_beta_function

from common import init_kmeans, softmax
from viz import plot_iteration

"""
Parameters:
    * maxIter: Max number of iterations
    * dataset: Dataset path
    * k: Number of clusters
    * verbose: Printing time, intermediate variational parameters, plots, ...
    * randomInit: Init assignations randomly or with Kmeans
    
Execution:
    python gmm_means_gavi.py -dataset data_k2_1000.pkl -k 2 -verbose 
"""

parser = argparse.ArgumentParser(description='CAVI in mixture og gaussians')
parser.add_argument('-maxIter', metavar='maxIter', type=int, default=300)
parser.add_argument('-dataset', metavar='dataset', type=str,
                    default='../../data/synthetic/2D/k2/data_k2_1000.pkl')
parser.add_argument('-k', metavar='k', type=int, default=2)
parser.add_argument('-verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('-randomInit', dest='randomInit', action='store_true')
parser.set_defaults(randomInit=False)
args = parser.parse_args()

K = args.k
VERBOSE = args.verbose
THRESHOLD = 1e-6


def update_lambda_pi(lambda_phi, alpha_o):
    """
    Update lambda_pi
    """
    return alpha_o + np.sum(lambda_phi, axis=0)


def update_lambda_phi(lambda_pi, lambda_m, lambda_beta,
                      lambda_phi, delta_o, xn, N, D):
    """
    Update lambda_phi
    """
    c1 = dirichlet_expectation(lambda_pi)
    for n in range(N):
        aux = np.copy(c1)
        for k in range(K):
            c2 = xn[n, :] - lambda_m[k, :]
            c3 = np.dot(delta_o, (xn[n, :] - lambda_m[k, :]).T)
            c4 = -1. / 2 * np.dot(c2, c3)
            c5 = D / (2. * lambda_beta[k])
            aux[k] += c4 - c5
        lambda_phi[n, :] = softmax(aux)
    return lambda_phi


def update_lambda_beta(lambda_phi, beta_o):
    """
    Update lambda_beta
    """
    return beta_o + np.sum(lambda_phi, axis=0)


def update_lambda_m(lambda_beta, lambda_phi, m_o, beta_o, xn, D):
    """
    Update lambda_m
    """
    d1 = np.tile(1. / lambda_beta, (D, 1)).T
    d2 = m_o * beta_o + np.dot(lambda_phi.T, xn)
    return d1 * d2


def elbo(xn, D, K, alpha, m_o, beta_o, delta_o,
         lambda_pi, lambda_m, lambda_beta, phi):
    """
    ELBO computation
    """
    lb = log_beta_function(lambda_pi)
    lb -= log_beta_function(alpha)
    lb += np.dot(alpha - lambda_pi, dirichlet_expectation(lambda_pi))
    lb += K / 2. * np.log(np.linalg.det(beta_o * delta_o))
    lb += K * D / 2.
    for k in range(K):
        a1 = lambda_m[k, :] - m_o
        a2 = np.dot(delta_o, (lambda_m[k, :] - m_o).T)
        a3 = beta_o / 2. * np.dot(a1, a2)
        a4 = D * beta_o / (2. * lambda_beta[k])
        a5 = 1 / 2. * np.log(np.linalg.det(lambda_beta[k] * delta_o))
        a6 = a3 + a4 + a5
        lb -= a6
        b1 = phi[:, k].T
        b2 = dirichlet_expectation(lambda_pi)[k]
        b3 = np.log(phi[:, k])
        b4 = 1 / 2. * np.log(np.linalg.det(delta_o) / (2. * math.pi))
        b5 = xn - lambda_m[k, :]
        b6 = np.dot(delta_o, (xn - lambda_m[k, :]).T)
        b7 = 1 / 2. * np.diagonal(np.dot(b5, b6))
        b8 = D / (2. * lambda_beta[k])
        lb += np.dot(b1, b2 - b3 + b4 - b7 - b8)
    return lb


def main():

    # Get data
    with open('{}'.format(args.dataset), 'r') as inputfile:
        data = pkl.load(inputfile)
        xn = data['xn']
    N, D = xn.shape

    if VERBOSE: init_time = time()

    # Priors
    alpha_o = [1.0] * K
    m_o = np.array([0.0, 0.0])
    beta_o = 0.01
    delta_o = np.zeros((D, D), long)
    np.fill_diagonal(delta_o, 1)

    # Variational parameters intialization
    lambda_phi = np.random.dirichlet(alpha_o, N) \
        if args.randomInit else init_kmeans(xn, N, K)
    lambda_beta = beta_o + np.sum(lambda_phi, axis=0)
    lambda_m = np.tile(1. / lambda_beta, (2, 1)).T * \
               (beta_o * m_o + np.dot(lambda_phi.T, xn))

    # Plot configs
    if VERBOSE:
        plt.ion()
        fig = plt.figure(figsize=(10, 10))
        ax_spatial = fig.add_subplot(1, 1, 1)
        circs = []
        sctZ = None

    # Inference
    n_iters = 0
    lbs = []
    for _ in range(args.maxIter):

        # Variational parameter updates
        lambda_pi = update_lambda_pi(lambda_phi, alpha_o)
        lambda_phi = update_lambda_phi(lambda_pi, lambda_m, lambda_beta,
                                       lambda_phi, delta_o, xn, N, D)
        lambda_beta = update_lambda_beta(lambda_phi, beta_o)
        lambda_m = update_lambda_m(lambda_beta, lambda_phi, m_o, beta_o, xn, D)

        # ELBO computation
        lb = elbo(xn, D, K, alpha_o, m_o, beta_o, delta_o,
                  lambda_pi, lambda_m, lambda_beta, lambda_phi)
        lbs.append(lb)

        if VERBOSE:
            print('\n******* ITERATION {} *******'.format(n_iters))
            print('lambda_pi: {}'.format(lambda_pi))
            print('lambda_beta: {}'.format(lambda_beta))
            print('lambda_m: {}'.format(lambda_m))
            print('lambda_phi: {}'.format(lambda_phi[0:9, :]))
            print('ELBO: {}'.format(lb))
            ax_spatial, circs, sctZ = plot_iteration(ax_spatial, circs, sctZ,
                                                     lambda_m, delta_o, xn,
                                                     n_iters, K)

        # Break condition
        improve = lb - lbs[n_iters - 1]
        if VERBOSE: print('Improve: {}'.format(improve))
        if (n_iters == (args.maxIter - 1)) \
                or (n_iters > 0 and 0 < improve < THRESHOLD):
            if VERBOSE and D == 2: plt.savefig('generated/plot.png')
            break

        n_iters += 1

    if VERBOSE:
        print('\n******* RESULTS *******')
        for k in range(K):
            print('Mu k{}: {}'.format(k, lambda_m[k, :]))
        final_time = time()
        exec_time = final_time - init_time
        print('Time: {} seconds'.format(exec_time))
        print('Iterations: {}'.format(n_iters))
        print('ELBOs: {}'.format(lbs))


if __name__ == '__main__': main()