"""
@Update: Fri Feb 02 2024 15:03:00
@MassylMoudoud: Recoded the function cutting_timecourse (change shape of input) to follow the convention

Copyright (c) 2024 University of Strasbourg
Author: Céline Meillier <meillier@unistra.fr>
Contributor(s) :Massyl Moudoud <mmoudoud@unistra.fr> , Vincent Mazet <vincent.mazet@unistra.fr>

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
Les fonction de ce fichier découpent les décours temporels par fenêtre glissante
(taille de fenêtre fixe, variable, phaseshift).

"""


def cutting_timecourse(U, window, slidind_step, standardisation=False):
    """Cuts timecourses to apply short time windowed correlations.

    Keyword arguments:
    U -- p x n array where p is the number of region and n is the length of timecourses.
    window -- m x 1 array containing the weighting window. If window is a scalar, it is considered as
    a rectangular window whose size is the parameter 'window'.
    sliding_step -- integer value
    standardisation -- boolean, if True, short time windowed timecourse are spatially standardised.
    """

    list_of_TCmatrices = []
    p, n = U.shape

    if isinstance(window, int):
        window = np.ones(window)
    lenW = len(window)

    windowMat = np.ones((p, lenW)) * window
    for i in range(0, n - lenW, slidind_step):
        # elementwise product (weighting signal)
        sig = windowMat[:, : min(n - i, lenW)] * U[:, i : min(i + lenW, n)]

        # standardisation of windowed signals
        if standardisation:
            Ui = ((sig - np.mean(sig, 0)) / np.std(sig, 0)).T
        else:
            Ui = sig
        list_of_TCmatrices.append(Ui)

    return list_of_TCmatrices
