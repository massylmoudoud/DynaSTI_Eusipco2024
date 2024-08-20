# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 2024  18:10:00

Copyright (c) 2024 University of Strasbourg
Author: Massyl Moudoud <mmoudoud@unistra.fr>
Contributor(s) : Céline Meillier <meillier@unistra.fr>, Vincent Mazet <vincent.mazet@unistra.fr>

This work has been supported by the ANR project DynaSTI: ANR-22-CE45-0008

This software is governed by the CeCILL  license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
"""

import mkl

print(mkl.get_max_threads())
mkl.set_num_threads(1)
print(mkl.get_max_threads())


import numpy as np
import multiprocessing as mp
import itertools
import sys

functionPath = "../"
sys.path.append(functionPath)

# from utile import *
# from Preprocessing import *
# from Construction_D import *
from Estimation_D.optimization_D_ADMM import estimate_D_ADMM
from Estimation_A.sunsal_tv import sunsal_tv_estimation
from joint_optimization_A_D_ADMM import joint_estimation_A_D

from generation_data import generate_data


"""
Script to optimize the hyperparameters of the ADMM and sunsalTV for the joint estimaton of A and D
"""


########################### Add noise to C #############################################


def make_noise(C, snr):
    """
    Function to add noise to the vectorized correlation matrix C to a given snr
    Inputs:
        C: R x T matrix, list of vectorized correlation matrices
        snr: desired Signal to Noise Ration (in dB)
    """
    P_c = np.sum(C**2) / np.prod(C.shape)
    sig_noise = P_c * 10 ** (-snr / 10)

    C_noise = C + np.random.default_rng().normal(0, sig_noise, size=C.shape)
    C_noise[C_noise >= 1] = 0.99
    C_noise[C_noise <= -1] = -0.99
    return C_noise


##################################################################################################
##################################################################################################


maxIter_D = 50


def eval_ADMM_D(rho, C_noise, C_VT, A_VT, D_VT, D_tilde, maxIter=maxIter_D):
    """
    Perform multiple evaluations of the search for the dictionary D using ADMM (function estimate_D_ADMM)
    with a grid search of the ADMM parameter rho
    the optimization is performed in parallel
    """

    D_est = estimate_D_ADMM(A_VT, C_noise, D_tilde, D_tilde, rho, maxIter)

    err_D = np.linalg.norm(D_est - D_VT, ord="fro")
    err_C = np.linalg.norm(C_VT - D_est @ A_VT, ord="fro")

    return {"rho": rho, "err_C": err_C, "err_D": err_D}


########################### Optimize A #################################################

max_iter_A = 250


def eval_sunsalTV_A(
    lambda_L1, lambda_TV, mu, C_noise, C_VT, A_VT, D_VT, maxIter_A=max_iter_A
):
    A_est, _, _ = sunsal_tv_estimation(
        D_VT,
        C_noise,
        lambda_l1=lambda_L1,
        lambda_TV=lambda_TV,
        mu=mu,
        AL_iters=maxIter_A,
    )

    err_A = np.linalg.norm(A_est - A_VT, ord="fro")
    err_C = np.linalg.norm(C_VT - D_VT @ A_est, ord="fro")

    return {
        "lambda_l1": lambda_L1,
        "lambda_TV": lambda_TV,
        "mu": mu,
        "err_C": err_C,
        "err_A": err_A,
    }


########################### Joint optimization #############################################

max_iter = 30


def eval_joint_optim_A_D(
    lambda_L1, lambda_TV, mu, C_noise, C_VT, A_VT, D_VT, D_tilde, max_iter=max_iter
):
    # set max iterations
    maxIter_A = 500
    maxIter_D = 50

    param_A = {
        "lambda_l1": lambda_L1,
        "AL_iters": maxIter_A,
        "mu": mu,
        "lambda_TV": lambda_TV,
    }

    param_D = {"rho": mu, "maxIter": maxIter_D}

    D_est, A_est = joint_estimation_A_D(C_noise, D_tilde, max_iter, param_D, param_A)
    # D_est, A_est = joint_estimation_A_D(C_noise, D_VT, D_tilde, max_iter,  param_D , param_A)

    err_D = np.linalg.norm(D_est - D_VT, ord="fro")
    err_A = np.linalg.norm(A_est - A_VT, ord="fro")
    err_C = np.linalg.norm(C_VT - D_est @ A_est, ord="fro")

    T = A_VT.shape[1]
    E = D_VT.shape[0]
    factor = np.sum(np.abs(D_est)) * T / (np.sum(np.abs(A_est)) * E)

    return {
        "lambda_l1": lambda_L1,
        "lambda_TV": lambda_TV,
        "mu": mu,
        "err_C": err_C,
        "err_A": err_A,
        "err_D": err_D,
        "factor": factor,
    }


#############################################################################################
#################################################################################################


def run_optim(realization):
    """
    input realization is the index of the realization, it is not used, the function map requires the function to be evaluated to take one argument
    """
    # Generate data
    C_noise, C_VT, A_VT, D_VT, D_tilde = generate_data(T=1000, snr=0)

    ##################################################################################################
    ##################################################################################################
    # Eval joint optimization A and D

    N_lambda_L1 = 10
    lambda_L1_min = 0
    lambda_L1_max = 0.3
    lambda_L1_vals = np.linspace(lambda_L1_min, lambda_L1_max, N_lambda_L1)

    N_lambda_TV = 10
    lambda_TV_min = 0
    lambda_TV_max = 1
    lambda_TV_vals = np.linspace(lambda_TV_min, lambda_TV_max, N_lambda_TV)

    N_mu = 1
    mu_min = 0.1
    mu_max = 10
    mu_vals = [1]  # np.linspace(mu_min, mu_max, N_mu )

    param_A_D = itertools.product(
        lambda_L1_vals,
        lambda_TV_vals,
        mu_vals,
        [C_noise],
        [C_VT],
        [A_VT],
        [D_VT],
        [D_tilde],
    )

    print(realization)

    res_A_D = [i for i in itertools.starmap(eval_joint_optim_A_D, param_A_D)]

    result_A_D = {"C_noise": C_noise, "C": C_VT, "A": A_VT, "result": res_A_D}
    return result_A_D


#################################################################################
########################### MAIN ################################################
#################################################################################


if __name__ == "__main__":
    mp.set_start_method("spawn")
    save_path = "./optimization_results/repeated_tests_simulation/"
    N_processes = 54

    N_realizations = 200

    chunksize = N_realizations // N_processes

    with mp.pool.Pool(N_processes) as p:
        result_A_D_realizations = [
            i for i in p.map(run_optim, range(N_realizations), chunksize=chunksize)
        ]

    np.save(save_path + "result_A_D_multirealizations", result_A_D_realizations)
