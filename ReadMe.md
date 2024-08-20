#Disclaimer
This code is a proof of concept and part of an ongoing work. It is provided for reproducing the results of a published paper.
__This code is NOT intented for production use__

# ANR DynaSTI: Paper Eusipco 2024
This repository contains the code of the methods published of the paper __Spatio-temporal model for dynamic functional connectivity in resting state fMRI analysis__ presented in the European Signal Processing Conference (EUSIPCO) 2024 held in Lyon, France.


# Aknowlegement
This work has been supported by the ANR (French researchagency "Agence Nationale de la Recherche") project DynaSTI: ANR-22-CE45-0008

# Code structure
The repository is stuctured as follows:
__src__: Contains the code of the methods to format the data, extract FCUs and display the results.

__scripts__: Contains the scripts to generate simulation data and to optimize model hyperparameters.

__notebooks__: The notebooks to reproduce the results and figures presented in the paper.

__data__: the data used in the experiments.


# Variable defintion:
In the code, we used the following naming convention for variable presented in the paper:

R: Number of ROIs (regions of interest)
E: Number of Pairs of ROIs ($R2 = \frac{R(R-1)}{2}$)
T: Time (or window index)
P: Number of FCUs (Functional Connectivity Units)


$\mathbf{TC}$ R x T Time courses of the R ROIs
$\mathbf{C}$ R x R correlation matrix
$\mathbf{Cvec}$ E x T list of vectorized correlation matrices of successive windows
$\mathbf{D}$ E x P Dictionary of FCUs 
$\mathbf{A}$ P x T matrix of activation of the FCUs in time




