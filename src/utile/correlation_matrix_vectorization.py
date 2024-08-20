# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 18:10:15 2022

@author: vport

@Update: Fri Feb 02 2024 11:55:00
@MassylMoudoud:
Recoded the function rebuild_C_matrix to adapt to new code
Modifed function build_C_vectors to enforce convention on matrices shapes


Copyright (c) 2024 University of Strasbourg
Author: Valentin Portmann
Contributor(s) : Massyl Moudoud <mmoudoud@unistra.fr>, Céline Meillier <meillier@unistra.fr>, Vincent Mazet <vincent.mazet@unistra.fr>

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

"""
These functions are useful from transforming a list of matrix into a list of vector, vis versa.
"""


def build_C_vectors(p, CMatrix_list):
    """Rebuild a list of correlation vectors from the list of
    matrix's correlations

    Keyword arguments:
    p = number of spatial areas (regions or networks)
    CMatrix_list = dictionnary of correlation represented by matrixes
    """

    # Number of useful data in the corr matrix
    vector_size = int(p * (p - 1) / 2)

    # Number of windows
    n_windows = len(CMatrix_list)

    # Initialization of the list of correlation's vectors
    CVector_list = np.zeros((vector_size, n_windows))

    # Mask of lower triangular matrix
    low_pos_useful_Data = np.ones((p, p))
    low_pos_useful_Data = np.tril(low_pos_useful_Data, k=-1)
    low_pos_useful_Data = low_pos_useful_Data != 0

    # Filling correlation Data
    for win, C in enumerate(CMatrix_list):
        CVector_list[:, win] = C[low_pos_useful_Data]

    return CVector_list


# ========================================================================
##MassylMoudoud: modified function to adapt to new code


def rebuild_C_matrix(Cvec, low=True):
    """Rebuild a list of correlation matrices from the list of
    vectorized correlation matrices (expressed as concatenated matrix or list of vectors)

    Inputs:
        Cvec: E x T matrix of concatenated vectorized lower triangular part of correlation matrices
                E: number of pairs of ROIs
                T: number of time poitns ( or windows)
        low : For backward compatibility
    Outputs:
        Clist: List of the T correlation matrices (each with shape R x R)
    """

    # If Cvec is a list, convert to array (for backward compatibility)
    if isinstance(Cvec, list):
        T = len(Cvec)
        E = len(Cvec[0])
        tmp = np.zeros((E, T))
        for i, C in enumerate(Cvec):
            tmp[:, i] = C
        Cvec = tmp
    elif not isinstance(Cvec, np.ndarray):
        raise TypeError("Incorrect type for Cvec, shold be numpy.ndarray or list")

    # number of pairs of regions and number of windows
    E, T = Cvec.shape
    # number of regions (ROIs)
    R = int((1 + np.sqrt(1 + 8 * E)) / 2)

    # Mask of lower triangular matrix
    low_pos_useful_Data = np.ones((R, R))
    if low:
        low_pos_useful_Data = np.tril(low_pos_useful_Data, k=-1)
    else:
        low_pos_useful_Data = np.triu(low_pos_useful_Data, k=-1)

    low_pos_useful_Data = low_pos_useful_Data != 0

    # Initialization of the list of correlation matrixes
    C_list = [np.zeros((R, R)) for win in range(T)]

    # Filling the list of correlation's matrixes
    for win in range(T):
        atom = Cvec[:, win]
        C_list[win][low_pos_useful_Data] = atom
        C_list[win] += C_list[win].T
        C_list[win] += np.eye(R)  # Put ones on the diag

    return C_list
