# -*- coding: utf-8 -*-
"""
Created on Tue  Apr 16 2024 09:40:00

@author: MassylMoudoud
"""


def load_fMRI_data(path):
    """
    Load the timecourse from the fMRI data

    Inputs:
        path: path to the folder containing the data
                The folder shold contain 2 subforlders:
                    ROIs_TC: Contains the time courses as npy files for each subject
                    ROIs_info: Contains the csv files for the list of ROI and the list of FCUs


    Output:
        TC_data: list of time courses for each subject
                each time course is an (R x T) size matrix: R is the number of ROIs, T is the number of time points.
        D_tilde: Support of dictionary as an E x P binary matrix: E number of pairs of ROI P number of FCUs
                 D_tilde define the stucture of the FCUs (which pairs of ROI compose each FCU)

    """

    raise NotImplementedError(
        "Due to the confidential aspect of the data, this function is not public\nPlease refere to the help to implment it for your data"
    )

    return TC_data, D_tilde
