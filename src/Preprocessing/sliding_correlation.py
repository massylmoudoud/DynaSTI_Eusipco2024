"""
@Update: Fri Feb 02 2024 15:03:00
@MassylMoudoud: modified the function list_of_corr_matrices (change shape of input) to follow the convention

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

from .cutting_timecourses_functions import cutting_timecourse

"""
Les fonction de ce fichier découpent les décours temporels par fenêtre glissante
(taille de fenêtre fixe, variable, phaseshift),
puis calculent sur chaque fenêtre la corrélation entre toutes les régions.

"""


def list_of_corr_matrices(U, lenW=30, step=1, standardisation=False):
    """Computes the short time windowed correlations.

    Keyword arguments:
    U -- p x n array where p is the number of region and n is the length of timecourses.
    lenW -- m x 1 array containing the weighting window. If window is a scalar, it is considered as 		a rectangular window whose size is the parameter 'window'.
    step -- integer value
    """
    list_of_TCmatrices = cutting_timecourse(
        U, lenW, step, standardisation=standardisation
    )
    N = len(list_of_TCmatrices)
    list_of_matrices = []
    for i in range(N):
        C = np.corrcoef(list_of_TCmatrices[i])
        list_of_matrices.append(C)
    return list_of_matrices
