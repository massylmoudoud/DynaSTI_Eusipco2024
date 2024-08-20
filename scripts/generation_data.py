"""
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
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from skimage.restoration import denoise_tv_chambolle
import scipy as sp


def state_generator(R, li):
    """
    R : nb of ROIs
    li : list of list of size 3 : [ROI1, ROI2, coef]
    """
    C = np.zeros((R, R))
    for l in li:
        ROI1 = l[0]
        ROI2 = l[1]
        coeff = l[2]
        C[ROI1, ROI2] = coeff
        C[ROI2, ROI1] = coeff
    return C


def network_generator(R, li, plot=False):
    ROI_name = [f"{i}" for i in range(R)]
    etat = state_generator(R, li)
    df = pd.DataFrame(data=etat, columns=ROI_name, index=ROI_name)
    G = nx.from_pandas_adjacency(df)
    weights = [G[u][v]["weight"] * 4 for u, v in G.edges()]
    M = G.number_of_edges()
    edge_colors = [G[u][v]["weight"] for u, v in G.edges()]

    if plot:
        plt.figure(figsize=(6, 6))
        nx.draw_networkx(
            G,
            with_labels=True,
            pos=position,
            width=weights,
            edge_color=edge_colors,
            edge_cmap=cmap,
            edge_vmin=-1,
            edge_vmax=1,
        )
    return etat, df, G


def get_dictionary():
    """
    returns the dictionary from hard coded FCUs (networks and subnetworks)
    """
    R = 10  # number of ROIs
    # Réseau 1a

    li1a = [
        [0, 1, 0.74],
        [0, 2, 0.8],
        [0, 3, 0.82],
        [1, 2, 0.72],
        [1, 3, 0.69],
        [2, 3, 0.83],
    ]
    etat1a, df1a, G1a = network_generator(R, li1a)

    # Réseau 1b

    li1b = [
        [5, 6, 0.65],
        [5, 7, 0.7],
        [5, 8, 0.67],
        [6, 7, 0.71],
        [6, 8, 0.69],
        [7, 8, 0.71],
    ]
    etat1b, df1b, G1b = network_generator(R, li1b)

    # Etat 1 (au sens des Kmeans)

    li1 = (
        li1a
        + li1b
        + [[9, 8, 0.13], [9, 5, 0.11], [4, 6, 0.14], [4, 5, 0.19], [4, 0, 0.13]]
    )
    etat1, df1, G1 = network_generator(R, li1)

    # Réseau 2a

    li2a = [
        [0, 1, 0.72],
        [0, 2, 0.81],
        [0, 3, 0.82],
        [1, 2, 0.67],
        [1, 3, 0.68],
        [2, 3, 0.79],
        [9, 0, 0.69],
        [9, 1, 0.65],
        [9, 2, 0.66],
        [9, 3, 0.67],
    ]

    etat2a, df2a, G2a = network_generator(R, li2a)

    # Réseau 2b

    li2b = [
        [6, 7, 0.56],
        [6, 8, 0.51],
        [7, 8, 0.76],
        [4, 5, 0.81],
        [4, 6, 0.52],
        [5, 6, 0.57],
        [4, 8, 0.17],
        [4, 7, 0.19],
        [5, 8, 0.23],
    ]
    etat2b, df2b, G2b = network_generator(R, li2b)

    # Réseau 2c

    li2c = [[6, 7, 0.66], [6, 8, 0.61], [7, 8, 0.76]]
    etat2c, df2c, G2c = network_generator(R, li2c)

    # Réseau 2d

    li2d = [[4, 5, 0.81], [4, 6, 0.52], [5, 6, 0.57]]
    etat2d, df2d, G2d = network_generator(R, li2d)

    # Etat 2

    li2 = li2a + li2b
    etat2, df2, G2 = network_generator(R, li2)

    # Réseau 3a

    li3a = [
        [0, 2, 0.81],
        [0, 3, 0.82],
        [0, 9, 0.79],
        [2, 3, 0.83],
        [2, 9, 0.7],
        [3, 9, 0.72],
    ]
    etat3a, df3a, G3a = network_generator(R, li3a)

    # Réseau 3b

    li3b = [
        [1, 4, 0.75],
        [1, 5, 0.68],
        [1, 6, 0.7],
        [4, 5, 0.81],
        [4, 6, 0.82],
        [5, 6, 0.77],
    ]
    etat3b, df3b, G3b = network_generator(R, li3b)

    # Etat 3
    # Composé des réseaux 3a et 3b, mais contient aussi des liaisons faibles entre plusieurs couples de régions pour favoriser la méthode des Kmeans.

    li3 = [
        [0, 2, 0.82],
        [0, 3, 0.82],
        [0, 9, 0.79],
        [2, 3, 0.83],
        [2, 9, 0.7],
        [3, 9, 0.72],
        [1, 4, 0.65],
        [1, 5, 0.68],
        [1, 6, 0.7],
        [4, 5, 0.81],
        [4, 6, 0.82],
        [5, 6, 0.77],
        [3, 7, 0.3],
        [4, 8, 0.2],
        [9, 7, 0.12],
        [2, 7, 0.18],
    ]
    etat3, df3, G3 = network_generator(R, li3)

    # Creating D
    # Here we consider each network and all its subnetworks as FCUs.
    # The matrix D contrans the weights of the edges of each pair of ROIs in a network or subnetwork

    network_list = ["1", "1a", "1b", "2", "2a", "2b", "2c", "2d", "3", "3a", "3b"]

    # dictionary of networks as keys and subnewtorks as values (expressed as indexes of the network_list)
    subnetwork_idx = {0: [1, 2], 3: [4, 5, 6, 7], 8: [9, 10]}

    N_networks = len(network_list)

    D = np.zeros((int(R * (R - 1) / 2), N_networks))

    idx = np.tril(np.ones((R, R)), -1) > 0

    for i, net in enumerate(network_list):
        net_list = eval("li" + net)
        C_net = np.zeros((R, R))
        for edge in net_list:
            C_net[edge[0], edge[1]] = edge[2]
            C_net[edge[1], edge[0]] = edge[2]

        D[:, i] = C_net[idx]

    D_tilde = np.int32(D > 0)

    return D, D_tilde, subnetwork_idx


def generate_data(T, snr):
    """
    generate sunthetic data according to the model C = DA and add noise to a given snr level
    Inputs:
        E: number of pairs of ROI
        P: Number of FCUs
        T: Number of time points
    """

    # get dictionary D

    D, D_tilde, subnetwork_idx = get_dictionary()

    E, N_networks = D.shape

    # initialization
    A = np.zeros((N_networks, T))

    # generate the activation probability at the start of the recording
    p = 0.5  # probability of a region to be active at the start of the recording
    initial_states = np.random.default_rng().binomial(1, p, size=N_networks)
    A[:, 0] = initial_states

    for i in range(N_networks):
        # generate the change time indicies
        durations = np.ceil(np.random.default_rng().uniform(40, 150, T // 100))
        indices = np.cumsum(durations)
        indices = indices[indices < T]

        for j in range(1, T):
            if np.isin(j, indices):  # at transition
                A[i, j] = 1 - A[i, j - 1]  # if 0 become 1 and if 1 becomes 0
            else:
                A[i, j] = A[i, j - 1]

    # check that when a network is active, none of its subnetworks is active simultanously

    for i in range(T):
        for n in subnetwork_idx.keys():
            # if all the subnetworks are active, activate the network
            if np.all(A[subnetwork_idx[n], i]):
                A[n, i] = 1

            # if network active deactivate all its subnetworks
            if A[n, i] == 1:
                A[subnetwork_idx[n], i] = 0

    # save the support of A
    A_tilde = A.copy()

    # remove too short activations
    active_min = 20

    transitions = np.abs(np.diff(A_tilde, axis=1, prepend=A_tilde[:, 0, np.newaxis]))
    for p in range(N_networks):  # for each FCU
        change_idx = np.where(transitions[p, :])[0]
        Nb_change = len(change_idx)
        for i in range(Nb_change - 1):  # discard last percept (stopped simulation)
            idx = change_idx[i]
            active_time = change_idx[i + 1] - change_idx[i]

            if active_time < active_min:
                A_tilde[p, idx : idx + active_time] = A_tilde[p, idx - 1]

    # Add noise to A
    noise = np.random.default_rng().normal(1, 0.5, size=(N_networks, T))

    A_noise = denoise_tv_chambolle(noise, weight=10, channel_axis=0) * A_tilde

    # enforce C<1

    def loss(A, A_noise):
        return np.linalg.norm(A - A_noise, ord=2) ** 2 / 2

    def jac(A, A_noise):
        return A - A_noise

    def hess(A, A_noise):
        return np.eye(A.shape[0])

    cons1 = {"type": "ineq", "fun": lambda A, D: 0.9 - D @ A, "args": (D,)}

    cons2 = {"type": "ineq", "fun": lambda A: A, "jac": lambda A: np.eye(A.shape[0])}

    opt = {"maxiter": 3, "disp": False}

    A = np.zeros((N_networks, T))

    for i in range(T):
        res = sp.optimize.minimize(
            loss,
            A_noise[:, i],
            args=A_noise[:, i],
            jac=jac,
            hess=None,
            constraints=[cons1, cons2],
            method="SLSQP",
            options=opt,
        )

        A[:, i] = res.x

    A = A * A_tilde

    # compute C
    C = D @ A

    # compute C_noise
    C_noise = make_noise(C, snr)

    return C_noise, C, A, D, D_tilde


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
