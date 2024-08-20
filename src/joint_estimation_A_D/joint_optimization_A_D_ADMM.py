# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 2024  16:24:00

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

import numpy as np

import sys

functionPath = "../"
sys.path.append(functionPath)

from Estimation_D.optimization_D_ADMM import estimate_D_ADMM
from Estimation_A.sunsal_tv import sunsal_tv_estimation


def joint_estimation_A_D(
    Cvec,
    D_tilde,
    max_iter,
    param_D,
    param_A,
    ret_err=False,
    C_VT=None,
    D_VT=None,
    A_VT=None,
):
    """
    Performs joint estimation of matrices A and D with the alternating optmization method

    Inputs:
        Cvec: R x T matrix of vectorized correlation matrices.
                R: Number of pairs of regions
                T: Number of windows
        D_tilde: R x P binary matrix. The support of the dictionary D
                P: Number of FCU
        max_iter: maximum number of iterations of the main optimization algorithm
        param_A: Dictionary of parameters to A optimization step with sunsalTV
                { "lambda_l1" , "AL_iters", "mu" ,  "lambda_TV" : lambda_TV , "verbose"}
                (see sunsal_tv_estimation for details)
        param_D: Dictionary of parameters to D optimization step with ADMM
                {"rho" , "maxIter" }
        ret_err: If True retruns the estimation error for each iteration
        C_VT   : Ground truth correlation matrix (when run on simulation data)
        D_VT   : Ground truth dictionary
        A_VT   : Ground truth activation matrix

    Outputs:
        D_est : Estimated Dictionary
        A_est : Estimated Activation A
        err   : Convergence of the reconstruction error  ||Cvec _ D_est * A_est||F
    """
    if ret_err:
        err = np.zeros((max_iter, 3))

    converged = False

    P = D_tilde.shape[1]  # number of FCUs

    # initialize D
    D_est = D_tilde

    # initialize A
    A_est, _, _ = sunsal_tv_estimation(D_est, Cvec, **param_A)

    for k in range(max_iter):
        # keep previous estimate
        A_prev = A_est.copy()
        D_prev = D_est.copy()

        # optimize D
        D_est = estimate_D_ADMM(
            A_est, Cvec, Delta_init=D_est, D_tilde=D_tilde, **param_D
        )

        # optimize A
        A_est, _, _ = sunsal_tv_estimation(D_est, Cvec, A_est, **param_A)

        # compoute reconstruction error
        C_est = D_est @ A_est
        if ret_err:
            if C_VT is not None:
                err[k, 0] = np.linalg.norm(C_est - C_VT, ord="fro")
            else:
                err[k, 0] = np.linalg.norm(C_est - Cvec, ord="fro")

            if D_VT is not None:
                err[k, 1] = np.linalg.norm(D_est - D_VT, ord="fro")
            else:
                err[k, 1] = np.linalg.norm(D_est - D_prev, ord="fro")

            if A_VT is not None:
                err[k, 2] = np.linalg.norm(A_est - A_VT, ord="fro")
            else:
                err[k, 2] = np.linalg.norm(A_est - A_prev, ord="fro")

        # check convergence
        if k % 10 == 0:
            print(k)

        if converged:
            err = err[:k, :]
            print("converged")
            break

    if ret_err:
        return D_est, A_est, err

    return D_est, A_est
