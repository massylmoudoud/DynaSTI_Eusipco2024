# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 2024  18:00:00

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

# imports
import numpy as np
import matplotlib.pyplot as plt


def plot_results(C_est, D_est, A_est, C, D=None, A=None, path=None):
    """
    Plot the results of the joint estimation of Dictionary D and activation matrix A

    """

    if path == None:
        save = False
    else:
        save = True

    # get data shapes
    R, T = C_est.shape
    P = D_est.shape[0]

    # dictionary D
    aspect_D = "auto"  # P/R
    if type(D) == type(None):  # D is not known
        plt.figure(figsize=(3, 3), layout="constrained")
        plt.imshow(D_est, aspect=aspect_D, cmap="hot")
        plt.title("Reconstruction")
        plt.grid(False)
        plt.colorbar()
    else:  # Ground truth D is known (simulation data)
        plt.figure(figsize=(10, 3), layout="constrained")
        plt.subplot(131)
        plt.imshow(D_est, aspect=aspect_D, cmap="hot")
        plt.title("Reconstruction")
        plt.grid(False)
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(D, aspect=aspect_D, cmap="hot")
        plt.title("Ground Truth D")
        plt.grid(False)
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(np.abs(D_est - D), aspect=aspect_D, cmap="hot")
        plt.title(r"Error $|\Delta - D_{gt}$")
        plt.grid(False)
        plt.colorbar()
        plt.suptitle("Dictionary D")

    if save:
        plt.savefig(path + "D.png")

    # Activation matrix A
    aspect_A = "auto"  # T/P
    if type(A) == type(None):  # A is not known
        plt.figure(layout="constrained", figsize=(17, 12))
        plt.subplot(311)
        plt.imshow(
            A_est, aspect=aspect_A, interpolation="nearest", cmap="magma"
        )  # , vmin = -1, vmax = 1)
        plt.grid(False)
        plt.colorbar()
        plt.title("Reconstruction")
        plt.suptitle("Activation matrix A")
    else:  # Ground truth A is known
        plt.figure(layout="constrained", figsize=(17, 12))
        plt.subplot(311)
        plt.imshow(
            A_est, aspect=aspect_A, interpolation="nearest", cmap="magma"
        )  # , vmin = -1, vmax = 1)
        plt.grid(False)
        plt.colorbar()
        plt.title("Reconstruction")
        plt.subplot(312)
        plt.imshow(
            A, aspect=aspect_A, interpolation="nearest", cmap="magma"
        )  # , vmin = -1, vmax = 1)
        plt.grid(False)
        plt.colorbar()
        plt.title("A")
        plt.subplot(313)
        plt.imshow(
            np.abs(A_est - A), aspect=aspect_A, interpolation="nearest", cmap="magma"
        )  # , vmin = -1, vmax = 1)
        plt.grid(False)
        plt.colorbar()
        plt.title(r"Error $|A_{est}- A|$")
        plt.suptitle("Activation matrix A")

    if save:
        plt.savefig(path + "A.png")

    # Correlation matrix C
    aspect_C = "auto"
    plt.figure(layout="constrained", figsize=(17, 12))
    plt.subplot(311)
    plt.imshow(
        C_est,
        aspect=aspect_C,
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.grid(False)
    plt.colorbar()
    plt.title("Reconstruction")
    plt.subplot(312)
    plt.imshow(
        C, aspect=aspect_C, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1
    )
    plt.grid(False)
    plt.colorbar()
    plt.title("C")
    plt.subplot(313)
    plt.imshow(
        np.abs(C_est - C),
        aspect=aspect_C,
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    plt.grid(False)
    plt.colorbar()
    plt.title(r"Error $|C_{est}- C|$")
    plt.suptitle("Correlation matrix C")

    if save:
        plt.savefig(path + "C.png")
