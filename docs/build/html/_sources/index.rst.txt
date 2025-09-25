.. NIRFASTerFF documentation master file, created by
   sphinx-quickstart on Tue Oct 15 14:16:53 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NIRFASTerFF's documentation!
=======================================
NIRFAST was originally developed in 2001 as a MATLAB-based package used to model near-infrared light propagation through tissue and reconstruct images of optical biomarkers. At its core, NIRFAST is tool that does finite element modeling of the medium in which the light progagates, and calculates the fluence field by solving the diffusion equation.

A fully overhauled version, titled NIRFASTer, was published in 2018, thanks to the work of Dr. Stanisław Wojtkiewicz. In the new version, GPU support was added to the key algorithms, giving the software a dramatic boost in performance. The CPU versions of the algorithms were re-implemented in C++ with multithreading enabled, and the performance was improved considerably.

In this version, now titled NIRFASTerFF (Fast and Furious), the entire toolbox is re-written with Python as its interfacing language, while fully inter-operatable with the original Matlab version. The algorithms, running on both GPU and CPU, are yet again improved for even better performance.

This manual is a detailed documentation of all APIs in the package. Please also refer to the demos to see how the package is used in difference applications.

Summary of the Functionalities
------------------------------

Mesh types supported: standard, fluorecence, and DCS

FEM solver calculates: CW fluence, FD fluence, TPSF, direct TR moments for standard and fluorecence mesh, and G1/g1 curve for DCS mesh

Analytical solution in semi-infinite medium for: CW/FD fluence, TPSF, and DCS G1/g1 curves

Jacobian matrices: CW for standard, fluorescence, and DCS mesh, and FD for standard and fluorescence mesh

Link to the Matlab Version
--------------------------
The original Matlab-based NIRFAST and NIRFASTer are still avaibable for download, but we will gradually drop our support for them.

https://github.com/nirfast-admin/NIRFAST

https://github.com/nirfaster/NIRFASTer

References
----------
If you use our package, please cite,

H. Dehghani, M.E. Eames, P.K. Yalavarthy, S.C. Davis, S. Srinivasan, C.M. Carpenter, B.W. Pogue, and K.D. Paulsen, "Near infrared optical tomography using NIRFAST: Algorithm for numerical model and image reconstruction," Communications in Numerical Methods in Engineering, vol. 25, 711-732 (2009) `doi:10.1002/cnm.1162 <https://doi.org/10.1002/cnm.1162>`_

API documentation
=================
.. toctree::
   :hidden:
   
   _autosummary/nirfasterff
