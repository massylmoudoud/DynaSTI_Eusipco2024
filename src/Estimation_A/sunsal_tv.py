# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 14:39:54 2022

@author: vport

Edited on Fri Jan 12 2024 16∶39∶39
@MassylMoudoud


Code translated from Matlab to python by Valentin prortman in 2022
Modified by Moudoud Massyl <mmoudoud@unistra.fr> in 2024

Code of the paper:

M.-D. Iordache, J. Bioucas-Dias, and A. Plaza, "Total variation spatial
regularization for sparse hyperspectral unmixing", IEEE Transactions on
Geoscience and Remote Sensing, vol. PP, no. 99, pp. 1-19, 2012.
------------------------------------------------------------------
Author: Jose Bioucas-Dias, January, 2010.


Modifications:

Jose Bioucas-Dias, July 2010:  -> Introduction of isotropic TV.


-------------------------------------------------------------------------

Copyright (January, 2011):        José Bioucas-Dias (bioucas@lx.it.pt)

SUNSAL_TV is distributed under the terms of
the GNU General Public License 2.0.

Permission to use, copy, modify, and distribute this software for
any purpose without fee is hereby granted, provided that this entire
notice is included in all copies of any software which is or includes
a copy or modification of this software and in all copies of the
supporting documentation for such software.
This software is being provided "as is", without any express or
implied warranty.  In particular, the authors do not make any
representation or warranty of any kind concerning the merchantability
of this software or its fitness for any particular purpose."
--------------------

"""

import numpy as np

"""
A method to estimate the activations through time of the different FCU,
 by reconstructing the correlations matrices on the dictionary of FCU.
"""


def sunsal_tv_estimation(
    M,
    Y,
    U=None,
    lambda_l1=0,
    lambda_TV=0,
    AL_iters=1000,
    mu=0.001,
    verbose=False,
    tol=None,
    return_list_U=False,
):
    """
    Résout:
        min 0.5*||MX-Y||^2 + i+(X) + is(X) + lambda_l1 * ||X||1_1 + lambda_tv * ||LX||1_1

    En résolvant:
        min 0.5*||V1-Y||^2 + i+(V2) + is(V3) + lambda_l1 * ||V4||1_1 + lambda_tv * ||V6||1_1
        avec:
            MU  = V1
            U   = V2=V3=V4=V5
            LV5 = V6

    Formats:
        M de taille (L, n)
        X de taille (n, N)
        Y de taille (L, N)

        L le nombre de variables (bandes spectrales par ex)
        n le nombre d'atomes/endmembers/material
        N la taille d'une image vectorisée.
        im_size[0]*im_size[1] = N

    Parameters
    ----------
    M : 2D array
        Dictionnaire.
    Y : 2D array
        Data.
    U : Initialization
    lambda_l1 : TYPE, optional
        l1 regularization parameter. The default is 0.
    lambda_TV : TYPE, optional
        TV regularization parameter. The default is 0.
    AL_iters : TYPE, optional
        Maximum number of AL iteration. The default is 1000.
    mu : float, optional
        AL weight. The default is 0.001.
    verbose : bool, optional
        Display only sunsal warnings. The default is False.

    Returns
    -------
    U : 2D numpy array
        X estimé.
    res : list
        Liste des residus par contrainte V.
    list_rmse : liste
        Liste de rmse par itération.

    """

    eps = np.finfo("float").eps

    # ====================================== Paramètrage
    # ----- Paramètres par défauts
    reg_l1 = 0
    reg_TV = 0
    reg_pos = 1

    if lambda_l1 > 0:
        reg_l1 = 1

    if lambda_TV > 0:
        reg_TV = 1

    # mixing matrix size
    [LM, n] = M.shape
    # data set size
    [L, N] = Y.shape
    if LM != L:
        raise NameError("mixing matrix M and data set y are inconsistent")

    if tol == None:
        tol = np.sqrt(N) * 1e-15

    # ====================================== Précalcul des opérateurs utilisés pour la TV
    # test for image size correctness
    if reg_TV > 0:
        # build handlers and necessary stuff
        # vertical difference operator
        FDv = np.zeros(N)
        FDv[0] = -1
        FDv[-1] = 1
        FDv = np.fft.fft(FDv)
        FDvH = np.conj(FDv)

        IL = 1 / (FDvH * FDv + 1)

    # ====================================== Définition de fonctions utiles pour les prox
    def Dv(x):
        return np.real(np.fft.ifft(np.fft.fft(x) * FDv))

    def DvH(x):
        return np.real(np.fft.ifft(np.fft.fft(x) * FDvH))

    def soft(x, T):
        T = T + eps
        y = np.maximum(np.abs(x) - T, 0)
        y = y / (y + T) * x
        return y

    # ====================================== Constantes
    # number of regularizers
    n_reg = reg_l1 + reg_pos + reg_TV

    IF = np.linalg.inv(np.matmul(np.transpose(M), M) + n_reg * np.identity(n))

    # ====================================== Initializations
    # MASSYL: Added option to initialize U (warm start)
    if U is None:
        # no intial solution supplied
        U = np.matmul(np.matmul(IF, np.transpose(M)), Y)

    # what regularizers ?
    #  0 - data term
    #  1 - positivity
    #  3 - l1
    #  4 - TV

    index = 0
    reg = []

    # initialize V variables
    V = []

    # initialize D variables (scaled Lagrange Multipliers)
    D = []

    # data term (always present)
    reg.append(0)  # regularizers
    V.append(np.matmul(M, U))  # V1
    D.append(np.zeros(Y.shape))  # Lagrange multipliers

    # next V
    index = index + 1
    # POSITIVITY
    if reg_pos == 1:
        reg.append(1)
        V.append(np.copy(U))
        D.append(np.zeros(U.shape))
        index = index + 1

    # l_{1,1}
    if reg_l1 == 1:
        reg.append(2)
        V.append(np.copy(U))
        D.append(np.zeros(U.shape))
        index = index + 1

    # TV
    # NOTE: V5, V6, D5, and D6 are represented as image planes
    if reg_TV == 1:
        # V5
        reg.append(3)
        V.append(np.copy(U))
        D.append(np.zeros(U.shape))

        # convert X into a cube
        U_im = np.reshape(np.transpose(U), (N, n))

        # V6 create two images per band (horizontal and vertical differences)
        # V[index+1] = np.zeros((n,2,im_size[0],im_size[1]))
        # D[index+1] = np.zeros((n,2,im_size[0],im_size[1]))
        V.append(np.zeros((n, N)))
        D.append(np.zeros((n, N)))

        for i in range(n):
            # build V6 image planes
            V[index + 1][i] = Dv(U_im[:, i])  # vertical differences
            # build D6 image planes
            D[index + 1][i] = np.zeros(N)  # vertical differences

        del U_im

    if verbose:
        print("0 - data term\n1 - positivity\n2 - l1\n3 - TV")

    if return_list_U:
        list_U = []
        list_res = []

    # MASSYL: Save the loss at each iteration
    loss = np.zeros(AL_iters)

    # ---------------------------------------------
    #  AL iterations - main body
    # ---------------------------------------------
    i = 1
    list_rmse = []
    res = np.inf * np.ones(n_reg + 1)
    while (i <= AL_iters) and (np.sum(np.abs(res)) > tol):
        # solve the quadratic step (all terms depending on U)
        Xi = np.matmul(np.transpose(M), V[0] + D[0])
        for j in range(1, n_reg + 1):
            Xi = Xi + V[j] + D[j]

        U = np.matmul(IF, Xi)
        if return_list_U:
            list_U.append(U)

        # Compute the Moreau proximity operators
        for j in range(n_reg + 1):
            # data term (V1)
            if reg[j] == 0:
                V[j] = 1 / (1 + mu) * (Y + mu * (np.matmul(M, U) - D[j]))

            # positivity (V2)
            if reg[j] == 1:
                V[j] = np.maximum(U - D[j], 0)

            # l1 norm  (V4)
            if reg[j] == 2:
                V[j] = soft(U - D[j], lambda_l1 / mu)

            # TV  (V5 and V6)
            if reg[j] == 3:
                # update V5: solves the problem:
                #    min 0.5*||L*V5-(V6+D7)||^2+0.5*||V5-(U-d5)||^2
                #      V5
                #
                # update V6: min 0.5*||V6-(L*V5-D6)||^2 + lambda_tv * |||V6||_{1,1}

                nu_aux = U - D[j]
                # convert nu_aux into image planes
                # convert X into a cube
                nu_aux5_im = np.transpose(nu_aux)

                V5_im = np.zeros((N, n))

                # compute V5 in the form of image planes
                for k in range(n):
                    # V5
                    V5_im[:, k] = np.real(
                        np.fft.ifft(
                            IL
                            * np.fft.fft(
                                DvH(V[j + 1][k] + D[j + 1][k]) + nu_aux5_im[:, k]
                            )
                        )
                    )

                    # V6
                    aux_v = Dv(V5_im[:, k])

                    # non-isotropic TV
                    V[j + 1][k] = soft(aux_v - D[j + 1][k], lambda_TV / mu)  # vertical

                    # update D6
                    D[j + 1][k] = D[j + 1][k] - (aux_v - V[j + 1][k])

                # convert V6 to matrix format
                V[j] = np.transpose(V5_im)

        # update Lagrange multipliers
        for j in range(n_reg + 1):
            if reg[j] == 0:
                D[j] = D[j] - (np.matmul(M, U) - V[j])
            else:
                D[j] = D[j] - (U - V[j])

        if return_list_U:
            for j in range(n_reg + 1):
                if reg[j] == 0:
                    res[j] = np.linalg.norm(np.matmul(M, U) - V[j], "fro")
                else:
                    res[j] = np.linalg.norm(U - V[j], "fro")
            list_res.append(np.copy(res))

        # compute residuals
        if np.mod(i, 10) == 1:
            st = ""
            for j in range(n_reg + 1):
                if reg[j] == 0:
                    res[j] = np.linalg.norm(np.matmul(M, U) - V[j], "fro")
                else:
                    res[j] = np.linalg.norm(U - V[j], "fro")
                st = st + " res(" + str(reg[j]) + ") = " + str(round(res[j], 3))

        # MASSYL: Save the evolution of the reconstruction error and evolution of A
        # loss[i-1] = np.linalg.norm(np.matmul(M,U)- Y,'fro')
        loss[i - 1] = (
            0.5 * np.linalg.norm(np.matmul(M, U) - Y, "fro") ** 2
            + lambda_l1 * np.sum(np.abs(U))
            + lambda_TV * np.sum(np.abs(np.diff(U, axis=1)))
        )

        if verbose:
            print("iter = ", i, "-", st)

        i = i + 1

    # MASSYL: keep only saved iterations
    loss = loss[:i]

    if return_list_U:
        return list_U, list_res

    # return U, res, list_rmse
    return U, res, loss
