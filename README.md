# nirfaster-FF

Public repository for the **Fast and Furious** version of NIRFASTer

- Version: 1.0.1
- Authors: Jiaming Cao, MILab@UoB
- License: TBD

This is the new NIRFASTer with Python interface, offering full functionality ranging from forward modeling, Jacobian calculating, analytical solutions, and simple visualization. The mesh format is fully compatible with the Matlab version.

## Requirements

The toolbox can run on Linux, Mac, and Windows. To use GPU acceleration, you will need to have a NVIDIA card with compute capability between `sm_52` (i.e. the GTX9xx series) and `sm_90a`. Please be noted that GPU support is unavailable on Mac platforms for now.

Only python versions 3.10 and 3.11 are supported for now.

Packages required:

- NumPy
- Scipy
- matplotlib

If you are using a Anaconda (or its alike) Python, they should already be available.

## Functionality

FEM-based mesh types and methods supported:

- Standard mesh
  - CW/FD data
  - TPSF/direct moments
  - Jacobian: CW/FD
- Fluorescence mesh
  - CW/FD data
  - TPSF/direct moments
  - Jacobian: CW/FD
- DCS mesh
  - G1/g1 curve
  - Jacobian

Analytical solutions provided (in semi-infinite space):

- CW/FD
- TPSF
- DCS G1/g1

Other components:

- Simple visualizers for 2D and 3D meshes
- A CGAL mesher (v6.0.1), enabling mesh creation from segmented volumes

## How to Install

1. Clone the main repo
2. Navigate to the *Release* section of the page, depending on your system and Python version, download the appropriate zip file(s) from the appropriate Release. Unzip the contents into the `nirfasterff/lib` folder
3. You should be good to go

Regardless of your setup, you will need to download the CPU library (cpu-*os*-python*). If your system is CUDA-enabled, you will *also* need to download the appropriate GPU library (gpu-*os*-python*), in addition.

**Special notes to Mac users**

Mac may throw a warning that the file is damaged and need to be moved to Trash. You can bypass this by using command

```bash
xattr -c <your_library>.so
```

## Citation

A paper on this package is currently in preperation. For now, if you are using our toolbox, please cite the original NIRFAST paper:

Dehghani, Hamid, et al. "Near infrared optical tomography using NIRFAST: Algorithm for numerical model and image reconstruction." Communications in numerical methods in engineering 25.6 (2009): 711-732. doi:10.1002/cnm.1162

## Documentation

Detailed API documentation can be found under `docs/build`, in both html and pdf (in folder "latex") formats. They are automatically compiled with Sphinx based on the docstrings in the Python codes.

If you wish to regenerate the documentation, navigate to the docs folder, and use command `build pdflatex` or `build html`.

## Tutorials

Detailed tutorials on how to use the package can be found under folder `tutorials`. They should give you a good start.

The head model is adapted from the examples in the [NeuroDOT toolbox](https://github.com/WUSTL-ORL/NeuroDOT)

## The Matlab version

You may still continue to use the old Matlab version, available [here](https://github.com/nirfaster/NIRFASTer). However, please be noted that we are slowly dropping support, which means that should you run into any bugs, we may not have the capacity to fix it for you.

## Changelog

1.0.1
- CGAL mesher separated from the CPU library as a standalone application
- Fixed a bug in gen_intmat, which leads to incorrect data.tomesh() result when xgrid and ygrid have different resolutions

## Some technical details

The reason for the Mac issue: Mac automatically attaches a quarantine attribute to downloaded files, and the marked files will be checked by the Gatekeeper. Somehow (file I/O, possibly), Apple's gatekeeper is not very happy about my code and refuses to run. This checking can be bypassed by manually removing the quarantine attribute. You can view this by `ls -l@`, and you should see the `com.apple.quarantine` thing.

Speed-critically functions are packed in precompiled libraries, nirfasterff_cpu and nirfasterff_cuda. The Linux and Mac versions are statically linked so there is only one file for each library plus the mesher, and no extra dependency is required. Only limited static linking could be used on Windows (e.g. the CUDA libraries), unfortunately, and consequently the necessary DLLs are also included.

#### Potential pitfalls:

- Python by default uses shallow copies, and sometimes fields in the mesh can be accidentally changed because of this. I tried to avoid this by explicitly using deep copies at various places, but undetected problems may still be there
- If a sliced array is fed into the C++ functions, they may throw a type error. This is because the sliced arrays may not be contiguous anymore. Using np.ascontiguousarray(*some_array*, dtype=*some_type*) will solve the problem. I tried to have this safeguard line in most of the high-level functions, but might have overlooked some
- When a C++ function takes a matrix argument, make sure it's actually a matrix. This is especially relevant when you try to feed it with an 1xN matrix. np.atleast2d() can help.
- Will not run on older linux distributions with GLIBC<3.25
