# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 2024  16:25:00

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


def estimate_D_ADMM(A, C, Delta_init, D_tilde, rho=1, maxIter=100, ret_err=False):
    """
    Solves the optimization problem
        min ..
     Using the ADMM algorithm
     Inputs:
        A : The P x T activation matrix
        C : The E x T vectorized correlation matrices
            E : the number of pair of ROIs
            T : The number of time windows
            P : The number of FCUs
        Delta_init : Initialization of the dictionary
        D_tilde : Support of the dictionary (define the nonzero entries in Delta)
        rho : ADMM learning rate
        maxIter : Number of iterations of the algorithm
        ret_err : If True, keep track and return estimation residuals
     Output:
        Delta : Estimated dictionary
        err : The 3 ADMM residuals:
                Primal residual:
                Dual residual
                Reonstruction error : ||C - Delta A||_F^2


    """
    # TODO: Write function help

    R, T = C.shape
    P = Delta_init.shape[1]

    # initialization
    Di = np.zeros((R, P, T))
    U = np.zeros((R, P, T))
    Delta = Delta_init

    # Iterations
    converged = False

    residuals = np.zeros((maxIter, 3))

    # Compute the inverses of A before the loop
    inv_ai = np.zeros((P, P, T))
    for i in range(T):
        inv_ai[:, :, i] = np.linalg.inv(
            A[:, i, np.newaxis] @ A[:, i, np.newaxis].T + rho * np.eye(P)
        )

    for k in range(maxIter):
        # step 1: update Di

        for i in range(T):
            Di[:, :, i] = (
                C[:, i, np.newaxis] @ A[:, i, np.newaxis].T + rho * (Delta - U[:, :, i])
            ) @ inv_ai[:, :, i]

        # keep the mean of Di
        D_bar = np.mean(Di, axis=2)

        # step 2: Update Delta
        # apply projection by using D_tilde as mask
        Delta_prev = Delta.copy()
        Delta = (np.mean(Di, axis=2) + np.mean(U, axis=2)) * D_tilde

        # enforce positivity of Delta
        Delta[Delta < 0] = 0

        # step 3: Update dual variable U
        for i in range(T):
            U[:, :, i] = U[:, :, i] + Di[:, :, i] - Delta

        if ret_err:
            # compute residuals
            residuals[k, 0] = np.sum(
                np.linalg.norm(Di - D_bar[:, :, np.newaxis], axis=(0, 1), ord="fro")
            )

            # based on Delta
            residuals[k, 1] = (
                np.linalg.norm(Delta - Delta_prev, ord="fro") ** 2
                / np.linalg.norm(Delta_prev) ** 2
            )

            # error in reconstructing C
            residuals[k, 2] = np.linalg.norm(C - Delta @ A, ord="fro")

        # check convergence
        # TODO

        if converged:
            residuals = residuals[:k, :]
            break

    if ret_err:
        return Delta, residuals
    else:
        return Delta
