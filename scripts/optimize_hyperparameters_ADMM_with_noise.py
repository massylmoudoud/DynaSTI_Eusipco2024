# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 2024  12:15:00

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

from Estimation_D.optimization_D_ADMM import estimate_D_ADMM
from Estimation_A.sunsal_tv import sunsal_tv_estimation
from joint_estimation_A_D.joint_optimization_A_D_ADMM import joint_estimation_A_D


from time import time


"""
Script to optimize the hyperparameters of the DMM and sunsalTV for the joint estimaton of A and D
"""

########################### Data ###############################################
# Load the simulation data
path = "data_simulation_A_D.npz"
data = np.load(path)
A_VT = data["A"]
D_VT = data["D"]

R, P = D_VT.shape
T = A_VT.shape[1]

D_tilde = np.int32(D_VT > 0)
C_VT = D_VT @ A_VT


#################################################################################
########################### FUNCTIONS ###########################################
#################################################################################

########################### Optimize D ############################################


maxIter_D = 50


# create the functions with the matrices as default parameters and options as variables
# Create a top level function to supply the default paramters to the ADMM
def eval_ADMM_D(
    rho,
    C_noise,
    A_VT=A_VT,
    C_VT=C_VT,
    D_VT=D_VT,
    Delta_init=D_tilde,
    D_tilde=D_tilde,
    maxIter=maxIter_D,
):
    """
    Perform multiple evaluations of the search for the dictionary D using ADMM (function estimate_D_ADMM)
    with a grid search of the ADMM parameter rho
    the optimization is performed in parallel
    """

    D_est = estimate_D_ADMM(A_VT, C_noise, Delta_init, D_tilde, rho, maxIter)

    err_D = np.linalg.norm(D_est - D_VT, ord="fro")
    err_C = np.linalg.norm(C_VT - D_est @ A_VT, ord="fro")

    return {"rho": rho, "err_C": err_C, "err_D": err_D}


########################### Optimize A #################################################

max_iter_A = 250


def eval_sunsalTV_A(
    lambda_L1,
    lambda_TV,
    mu,
    C_noise,
    maxIter_A=max_iter_A,
    A_VT=A_VT,
    C_VT=C_VT,
    D_VT=D_VT,
):
    start_time = time()

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

    print("it took", time() - start_time)
    return {
        "lambda_l1": lambda_L1,
        "lambda_TV": lambda_TV,
        "mu": mu,
        "err_C": err_C,
        "err_A": err_A,
    }


########################### Joint optimization #############################################

max_iter = 25


def eval_joint_optim_A_D(
    lambda_L1,
    lambda_TV,
    mu,
    C_noise,
    max_iter=max_iter,
    A_VT=A_VT,
    C_VT=C_VT,
    D_VT=D_VT,
    D_tilde=D_tilde,
):
    start_time = time()

    # set max iterations
    maxIter_A = 250
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

    print("it took", time() - start_time)
    return {
        "lambda_l1": lambda_L1,
        "lambda_TV": lambda_TV,
        "mu": mu,
        "err_C": err_C,
        "err_A": err_A,
        "err_D": err_D,
    }


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


#################################################################################
########################### MAIN ###########################################
#################################################################################

if __name__ == "__main__":
    start_time = time()
    mp.set_start_method("spawn")

    eval_D = False
    eval_A = False
    eval_A_D = True

    save_path = "./optimization_results/"
    N_processes = 54
    snr_vals = [0, 10]

    for snr in snr_vals:
        # Add noise to C
        C_noise = make_noise(C_VT, snr)

        ########################### Optimize D #################################################
        # Eval ADMM on D
        if eval_D:
            N_rho = 100
            rho_min = 0.001
            rho_max = 4
            rho_vals = np.linspace(rho_min, rho_max, N_rho)

            param_D = itertools.product(rho_vals, [C_noise])

            chunksize = N_rho // N_processes

            with mp.pool.Pool(N_processes) as p:
                result_D = [
                    i for i in p.starmap(eval_ADMM_D, param_D, chunksize=chunksize)
                ]

            # save result
            np.save(save_path + f"result_D_SNR_{snr}", result_D)

        ########################### Optimize A #################################################
        # Eval SunsalTV for A
        if eval_A:
            N_lambda_L1 = 50
            lambda_L1_min = 0
            lambda_L1_max = 0.8
            lambda_L1_vals = np.linspace(lambda_L1_min, lambda_L1_max, N_lambda_L1)

            N_lambda_TV = 50
            lambda_TV_min = 0
            lambda_TV_max = 2
            lambda_TV_vals = np.linspace(lambda_TV_min, lambda_TV_max, N_lambda_TV)

            N_mu = 2
            mu_min = 0.1
            mu_max = 10
            mu_vals = [0.1, 1]  # np.linspace(mu_min, mu_max, N_mu )

            param_A = itertools.product(
                lambda_L1_vals, lambda_TV_vals, mu_vals, [C_noise]
            )

            chunksize = (N_mu * N_lambda_TV * N_lambda_L1) // N_processes

            with mp.pool.Pool(N_processes) as p:
                result_A = [
                    i for i in p.starmap(eval_sunsalTV_A, param_A, chunksize=chunksize)
                ]

            # save result
            np.save(save_path + f"result_A_SNR_{snr}", result_A)

        ########################### Joint optimization #############################################

        # Eval joint optimization A and D
        if eval_A_D:
            N_lambda_L1 = 50
            lambda_L1_min = 0
            lambda_L1_max = 0.5
            lambda_L1_vals = np.linspace(lambda_L1_min, lambda_L1_max, N_lambda_L1)

            N_lambda_TV = 50
            lambda_TV_min = 0
            lambda_TV_max = 2
            lambda_TV_vals = np.linspace(lambda_TV_min, lambda_TV_max, N_lambda_TV)

            N_mu = 1
            mu_min = 0.1
            mu_max = 10
            mu_vals = [1]  # np.linspace(mu_min, mu_max, N_mu )

            # NOTE: the ADMM parameter is set equally for A (called mu) and D (called rho)
            param_A_D = itertools.product(
                lambda_L1_vals, lambda_TV_vals, mu_vals, [C_noise]
            )

            chunksize = (N_mu * N_lambda_TV * N_lambda_L1) // N_processes

            with mp.pool.Pool(N_processes) as p:
                result_A_D = [
                    i
                    for i in p.starmap(
                        eval_joint_optim_A_D, param_A_D, chunksize=chunksize
                    )
                ]

            # save result
            np.save(save_path + f"result_A_D_SNR_{snr}", result_A_D)

            print("finished at", start_time - time())

    print("finished at", time())
