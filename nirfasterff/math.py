"""
Some low-level functions used by the forward solvers.

It is usually unnecessary to use them directly and caution must be excercised, as many of them interact closely with the C++ libraries and can cause crashes if used incorrectly
"""
import numpy as np
from nirfasterff import utils
import copy
from scipy import sparse

def gen_mass_matrix(mesh, omega, solver = utils.get_solver(), GPU = -1):
    """
    Calculate the MASS matrix, and return the coordinates in CSR format.
    
    The current Matlab version outputs COO format, so the results are NOT directly compatible
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU

    Parameters
    ----------
    mesh : nirfasterff.base.stndmesh
        the mesh used to calculate the MASS matrix.
    omega : double
        modulation frequency, in radian.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    GPU : int, optional
        GPU selection. -1 for automatic, 0, 1, ... for manual selection on multi-GPU systems. The default is -1.

    Raises
    ------
    RuntimeError
        if both CUDA and CPU versions fail.
    TypeError
        if 'solver' is not 'CPU' or 'GPU'.

    Returns
    -------
    csrI: int32 NumPy vector, zero-based
        I indices of the MASS matrix, in CSR format. Size (NNodes,)
    csrJ: int32 NumPy vector, zero-based
        J indices of the MASS matrix, in CSR format. Size (nnz(MASS),)
    csrV: float64 or complex128 NumPy vector
        values of the MASS matrix, in CSR format. Size (nnz(MASS),)

    """
    
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [csrI, csrJ, csrV] = utils.cudalib.gen_mass_matrix(mesh.nodes, mesh.elements, mesh.bndvtx, mesh.mua, mesh.kappa, mesh.ksi, mesh.c, omega, GPU)
            if omega==0:
                csrV = np.real(csrV)
        except:
            print('Warning: GPU code failed. Rolling back to CPU code')
            try:
                [csrI, csrJ, csrV] = utils.cpulib.gen_mass_matrix(mesh.nodes, mesh.elements, mesh.bndvtx, mesh.mua, mesh.kappa, mesh.ksi, mesh.c, omega)
                if omega==0:
                    csrV = np.real(csrV)    
            except:
                raise RuntimeError('Error: couldn''t generate mass matrix')
    elif solver.lower()=='cpu':
        try:
            [csrI, csrJ, csrV] = utils.cpulib.gen_mass_matrix(mesh.nodes, mesh.elements, mesh.bndvtx, mesh.mua, mesh.kappa, mesh.ksi, mesh.c, omega)
            if omega==0:
                csrV = np.real(csrV) 
        except:
            raise RuntimeError('Error: couldn''t generate mass matrix')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return csrI, csrJ, np.ascontiguousarray(csrV)

def get_field_CW(csrI, csrJ, csrV, qvec, opt = utils.SolverOptions(), solver=utils.get_solver()):
    """
    Call the Preconditioned Conjugate Gradient solver with FSAI preconditioner. For CW data only.
    
    The current Matlab version uses COO format input, so they are NOT directly compatible
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
    
    On GPU, the algorithm first tries to solve for all sources simultaneously, but this can fail due to insufficient GPU memory.
    
    If this is the case, it will generate a warning and solve the sources one by one. The latter is not as fast, but requires much less memory.
    
    On CPU, the algorithm only solves the sources one by one.

    Parameters
    ----------
    csrI : int32 NumPy vector, zero-based
        I indices of the MASS matrix, in CSR format.
    csrJ : int32 NumPy vector, zero-based
        J indices of the MASS matrix, in CSR format.
    csrV : double NumPy vector
        values of the MASS matrix, in CSR format.
    qvec : double NumPy array, or Scipy CSC sparse matrix
        The source vectors. i-th column corresponds to source i. Size (NNode, NSource)
        
        See :func:`~nirfasterff.math.gen_sources()` for details.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if MASS matrix and source vectors are not both real, or if solver is not 'CPU' or 'GPU'.
    RuntimeError
        if both GPU and CPU solvers fail.

    Returns
    -------
    phi: double NumPy array
        Calculated fluence at each source. Size (NNodes, Nsources)
    info: nirfasterff.utils.ConvergenceInfo 
        convergence information of the solver.
        
        See :func:`~nirfasterff.utils.ConvergenceInfo` for details
    
    See Also
    -------
    :func:`~nirfasterff.math.gen_mass_matrix()`

    """
    
    if not (np.isreal(csrV).all() and np.isrealobj(qvec)):
        raise TypeError('MASS matrix and qvec should be both real')
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [phi, info] = utils.cudalib.get_field_CW(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                [phi, info] = utils.cpulib.get_field_CW(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
            except:
                raise RuntimeError('Error: solver failed')
    elif solver.lower()=='cpu':
        try:
            [phi, info] = utils.cpulib.get_field_CW(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
        except:
            raise RuntimeError('Error: solver failed')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return phi, utils.ConvergenceInfo(info)

def get_field_FD(csrI, csrJ, csrV, qvec, opt = utils.SolverOptions(), solver=utils.get_solver()):
    """
    Call the Preconditioned BiConjugate Stablized solver with FSAI preconditioner. 
    
    This is designed for FD data, but can also work for CW is an all-zero imaginary part is added to the MASS matrix and source vectors.
    
    The current Matlab version uses COO format input, so they are NOT directly compatible
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
    
    On GPU, the algorithm first tries to solve for all sources simultaneously, but this can fail due to insufficient GPU memory.
    
    If this is the case, it will generate a warning and solve the sources one by one. The latter is not as fast, but requires much less memory.
    
    On CPU, the algorithm only solves the sources one by one.

    Parameters
    ----------
    csrI : int32 NumPy vector, zero-based
        I indices of the MASS matrix, in CSR format.
    csrJ : int32 NumPy vector, zero-based
        J indices of the MASS matrix, in CSR format.
    csrV : complex double NumPy vector
        values of the MASS matrix, in CSR format.
    qvec : complex double NumPy array, or Scipy CSC sparse matrix
        The source vectors. i-th column corresponds to source i. Size (NNode, NSource)
        
        See :func:`~nirfasterff.math.gen_sources()` for details.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if MASS matrix and source vectors are not both complex, or if solver is not 'CPU' or 'GPU'.
    RuntimeError
        if both GPU and CPU solvers fail.

    Returns
    -------
    phi: complex double NumPy array
        Calculated fluence at each source. Size (NNodes, Nsources)
    info: nirfasterff.utils.ConvergenceInfo 
        convergence information of the solver.
        
        See :func:`~nirfasterff.utils.ConvergenceInfo` for details
    
    See Also
    -------
    :func:`~nirfasterff.math.gen_mass_matrix()`

    """
    
    if not np.all(np.iscomplex(csrV).all() and np.iscomplexobj(qvec)):
        raise TypeError('MASS matrix and qvec should be both complex')
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [phi, info] = utils.cudalib.get_field_FD(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                [phi, info] = utils.cpulib.get_field_FD(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
            except:
                raise RuntimeError('Error: solver failed')
    elif solver.lower()=='cpu':
        try:
            [phi, info] = utils.cpulib.get_field_FD(csrI, csrJ, csrV, qvec, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
        except:
            raise RuntimeError('Error: solver failed')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return phi, utils.ConvergenceInfo(info)

def get_field_TR(csrI, csrJ, csrV, qvec, dt, max_step, opt = utils.SolverOptions(), solver=utils.get_solver()):
    """
    Call the Preconditioned Conjugate Gradient solver with FSAI preconditioner. Calculates TPSF data
    
    NOT interchangeable with the current MATLAB version
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
    
    On both GPU and CPU, the algorithm solves the sources one by one

    Parameters
    ----------
    csrI : int32 NumPy vector, zero-based
        I indices of the MASS matrices, in CSR format.
    csrJ : int32 NumPy vector, zero-based
        J indices of the MASS matrices, in CSR format.
    csrV : complex double NumPy vector
        values of the MASS matrices, in CSR format. 
        
        This is calculated using gen_mass_matrix with omega=1. The real part coincides with K+C, and the imaginary part coincides with -M.
        
        See references for details
    qvec : double NumPy array, or Scipy CSC sparse matrix
        The source vectors. i-th column corresponds to source i. Size (NNode, NSource)
        
        See :func:`~nirfasterff.math.gen_sources()` for details.
    dt : double 
        time step size, in seconds.
    max_step : int32
        total number of time steps.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if csrV is not complex, or if qvec is not real, or if solver is not 'CPU' or 'GPU'.
    RuntimeError
        if both GPU and CPU solvers fail.

    Returns
    -------
    phi: double NumPy array
        Calculated TPSF at each source. Size (NNodes, Nsources*max_step), structured as,
        
        [src0_step0, src1_step0,...,src0_step1, src1_step1,...]
    info: nirfasterff.utils.ConvergenceInfo 
        convergence information of the solver.
        
        Only the convergence info of the last time step is returned.
        
        See :func:`~nirfasterff.utils.ConvergenceInfo` for details
    
    See Also
    -------
    :func:`~nirfasterff.math.gen_mass_matrix()`
    
    References
    -------
    Arridge et al., Med. Phys,, 1993. doi:10.1118/1.597069

    """
   
    if not np.all(np.iscomplex(csrV).all()):
        raise TypeError('Missing M matrix')
    if not np.isrealobj(qvec):
        raise TypeError('qvec must be real')
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    Sr = np.ascontiguousarray(csrV.real)
    Si = np.ascontiguousarray(csrV.imag)
    # MASS matrix calculated with w=1, therefore real part of csrV corresponds to K+C, and imag part corresponds to -M
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [phi, info] = utils.cudalib.get_field_TR(csrI, csrJ, 0.5*Sr-Si/dt, -(0.5*Sr+Si/dt), qvec, max_step, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                [phi, info] = utils.cpulib.get_field_TR(csrI, csrJ, 0.5*Sr-Si/dt, -(0.5*Sr+Si/dt), qvec, max_step, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
            except:
                raise RuntimeError('Error: solver failed')
    elif solver.lower()=='cpu':
        try:
            [phi, info] = utils.cpulib.get_field_TR(csrI, csrJ, 0.5*Sr-Si/dt, -(0.5*Sr+Si/dt), qvec, max_step, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
        except:
            raise RuntimeError('Error: solver failed')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return phi/dt, utils.ConvergenceInfo(info)

def get_field_TRmoments(csrI, csrJ, csrV, qvec, max_moment, opt = utils.SolverOptions(), solver=utils.get_solver()):
    """
    Call the Preconditioned Conjugate Gradient solver with FSAI preconditioner. Directly calculates moments of TR data using Mellin transform
    
    NOT interchangeable with the current MATLAB version
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
    
    On both GPU and CPU, the algorithm solves the sources one by one

    Parameters
    ----------
    csrI : int32 NumPy vector, zero-based
        I indices of the MASS matrices, in CSR format.
    csrJ : int32 NumPy vector, zero-based
        J indices of the MASS matrices, in CSR format.
    csrV : complex double NumPy vector
        values of the MASS matrices, in CSR format. 
        
        This is calculated using gen_mass_matrix with omega=1. The real part coincides with K+C, and the imaginary part coincides with -B.
        
        See references for details
    qvec : double NumPy array, or Scipy CSC sparse matrix
        The source vectors. i-th column corresponds to source i. Size (NNode, NSource)
    max_moment : int32
        max order of moments to calculate. That is, 0th, 1st, 2nd, .., max_moments-th will be calculated.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if csrV is not complex, or if qvec is not real, or if solver is not 'CPU' or 'GPU'.
    RuntimeError
        if both GPU and CPU solvers fail.

    Returns
    -------
    phi: double NumPy array
        Calculated Mellin transform at each source. Size (NNodes, Nsources*(max_moment+1)), structured as,
        
        [src0_m0, src1_m0,...,src0_m1, src1_m1,...]
        
    info: nirfasterff.utils.ConvergenceInfo 
        convergence information of the solver.
        
        Only the convergence info of the highest order moments is returned.
        
        See :func:`~nirfasterff.utils.ConvergenceInfo` for details
    
    See Also
    -------
    :func:`~nirfasterff.math.gen_mass_matrix()`
    
    References
    -------
    Arridge and Schweiger, Applied Optics, 1995. doi:10.1364/AO.34.002683

    """
    
    if not np.all(np.iscomplex(csrV).all()):
        raise TypeError('Missing B matrix')
    if not np.isrealobj(qvec):
        raise TypeError('qvec must be real')
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    Sr = np.ascontiguousarray(csrV.real)
    Si = np.ascontiguousarray(csrV.imag)
    # MASS matrix calculated with w=1, therefore real part of csrV corresponds to K+C, and imag part corresponds to -B
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [phi, info] = utils.cudalib.get_field_TR_moments(csrI, csrJ, Sr, -Si, qvec, max_moment, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                [phi, info] = utils.cpulib.get_field_TR_moments(csrI, csrJ,  Sr, -Si, qvec, max_moment, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
            except:
                raise RuntimeError('Error: solver failed')
    elif solver.lower()=='cpu':
        try:
            [phi, info] = utils.cpulib.get_field_TR_moments(csrI, csrJ,  Sr, -Si, qvec, max_moment, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
        except:
            raise RuntimeError('Error: solver failed')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return phi, utils.ConvergenceInfo(info)

def get_field_TRFL(csrI, csrJ, csrV, qvec_m, dt, max_step, opt = utils.SolverOptions(), solver=utils.get_solver()):
    """
    Call the Preconditioned Conjugate Gradient solver with FSAI preconditioner. Calculates the TPSFs of fluorescence emission given the TPSFs of excitation
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
    
    On both GPU and CPU, the algorithm solves the sources one by one

    Parameters
    ----------
    csrI : int32 NumPy vector, zero-based
        I indices of the MASS matrices at emission wavelength, in CSR format.
    csrJ : int32 NumPy vector, zero-based
        J indices of the MASS matrices at emission wavelength, in CSR format.
    csrV : complex double NumPy vector
        values of the MASS matrices at emission wavelength, in CSR format. 
        
        This is calculated using gen_mass_matrix with omega=1.
    qvec_m : double NumPy array
        TPSF of the excitation convolved with decay, and multiplied by the FEM matrix. Size (NNodes, NSources*NTime), structured as,
        
        [src0_step0, src1_step0,...,src0_step1, src1_step1,...]
    dt : double 
        time step size, in seconds.
    max_step : int32
        total number of time steps. It should match exactly with the number of steps in the excitation data
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if csrV is not complex, or if phix is not real, or if solver is not 'CPU' or 'GPU'.
    RuntimeError
        if both GPU and CPU solvers fail.

    Returns
    -------
    phi: double NumPy array
        Calculated TPSF at each source of fluorescence emission. Size (NNodes, Nsources*max_step), structured as,
        
        [src0_step0, src1_step0,...,src0_step1, src1_step1,...]
    info: nirfasterff.utils.ConvergenceInfo 
        convergence information of the solver.
        
        Only the convergence info of the last time step is returned.
        
        See :func:`~nirfasterff.utils.ConvergenceInfo` for details
    
    See Also
    -------
    :func:`~nirfasterff.math.gen_mass_matrix()`, :func:`~nirfasterff.math.get_field_TR()`

    """

    if not np.all(np.iscomplex(csrV).all()):
        raise TypeError('Missing M matrix')
    if not np.isrealobj(qvec_m):
        raise TypeError('qvec_m must be real')
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    Sr = np.ascontiguousarray(csrV.real)
    Si = np.ascontiguousarray(csrV.imag)
    # MASS matrix calculated with w=1, therefore real part of csrV corresponds to K+C, and imag part corresponds to -M
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [phi, info] = utils.cudalib.get_field_TRFL(csrI, csrJ, 0.5*Sr-Si/dt, -(0.5*Sr+Si/dt), qvec_m, max_step, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                [phi, info] = utils.cpulib.get_field_TRFL(csrI, csrJ, 0.5*Sr-Si/dt, -(0.5*Sr+Si/dt), qvec_m, max_step, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
            except:
                raise RuntimeError('Error: solver failed')
    elif solver.lower()=='cpu':
        try:
            [phi, info] = utils.cpulib.get_field_TRFL(csrI, csrJ, 0.5*Sr-Si/dt, -(0.5*Sr+Si/dt), qvec_m, max_step, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
        except:
            raise RuntimeError('Error: solver failed')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return phi/dt, utils.ConvergenceInfo(info)

def get_field_TRFLmoments(csrI, csrJ, csrV, csrV2, mx, gamma, tau, max_moment, opt = utils.SolverOptions(), solver=utils.get_solver()):
    """
    Call the Preconditioned Conjugate Gradient solver with FSAI preconditioner. Directly calculates moments of re-emission using Mellin transform, given the moments of excitation

    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.
    
    On both GPU and CPU, the algorithm solves the sources one by one

    Parameters
    ----------
    csrI : int32 NumPy vector, zero-based
        I indices of the MASS matrices at emission wavelength, in CSR format.
    csrJ : int32 NumPy vector, zero-based
        J indices of the MASS matrices at emission wavelength, in CSR format.
    csrV : complex double NumPy vector
        values of the MASS matrices at emission wavelength, in CSR format. 
        
        This is calculated using gen_mass_matrix with omega=1.
    csrV2 : double NumPy vector
        values of the FEM integration matrix, in CSR format. 
        
        This is calculated using gen_mass_matrix with omega=0, mua=1, kappa=0, and no boundary nodes.
    mx : double NumPy array
        moments of the excitation. Size (NNodes, Nsources*(max_moment+1)), structured as,
        
        [src0_m0, src1_m0,...,src0_m1, src1_m1,...]
    gamma : double NumPy array
        defined as mesh.eta*mesh.muaf.
    tau : double NumPy array
        decay factor, as defined in mesh.tau.
    max_moment : int32
        max order of moments to calculate. That is, 0th, 1st, 2nd, .., max_moments-th will be calculated.
        
        This should match exact with the max_moment of the excitation
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if csrV is not complex, or if mx is not real, or if solver is not 'CPU' or 'GPU'.
    RuntimeError
        if both GPU and CPU solvers fail.

    Returns
    -------
    phi: double NumPy array
        Calculated Mellin transform of fluorecence emission at each source. Size (NNodes, Nsources*(max_moment+1)), structured as,
        
        [src0_m0, src1_m0,...,src0_m1, src1_m1,...]
        
    info: nirfasterff.utils.ConvergenceInfo 
        convergence information of the solver.
        
        Only the convergence info of the highest order moments is returned.
        
        See :func:`~nirfasterff.utils.ConvergenceInfo` for details
    
    See Also
    -------
    :func:`~nirfasterff.math.gen_mass_matrix()`, :func:`~nirfasterff.math.get_field_TRmoments()`

    """
    
    if not np.all(np.iscomplex(csrV).all()):
        raise TypeError('Missing B matrix')
    if not np.isrealobj(mx):
        raise TypeError('mx must be real')
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
        
    Sr = np.ascontiguousarray(csrV.real)
    Si = np.ascontiguousarray(csrV.imag)
    # MASS matrix calculated with w=1, therefore real part of csrV corresponds to K+C, and imag part corresponds to -M
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            [phi, info] = utils.cudalib.get_field_TRFL_moments(csrI, csrJ, Sr, -Si, csrV2, mx, gamma, tau, max_moment, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, opt.GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                [phi, info] = utils.cpulib.get_field_TRFL_moments(csrI, csrJ,  Sr, -Si, csrV2, mx, gamma, tau, max_moment, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
            except:
                raise RuntimeError('Error: solver failed')
    elif solver.lower()=='cpu':
        try:
            [phi, info] = utils.cpulib.get_field_TRFL_moments(csrI, csrJ,  Sr, -Si, csrV2, mx, gamma, tau, max_moment, opt.max_iter, opt.AbsoluteTolerance, opt.RelativeTolerance, opt.divergence, utils.get_nthread())  
        except:
            raise RuntimeError('Error: solver failed')
    else:
        raise TypeError('Error: Solver should be ''GPU'' or ''CPU''')
        
    return phi, utils.ConvergenceInfo(info)

def gen_sources(mesh):
    """
    Calculate the source vectors (point source only) for the sources in mesh.source field

    Parameters
    ----------
    mesh : NIRFASTer mesh type
        mesh used to calculate the source vectors. Source information is also defined here.

    Returns
    -------
    qvec : complex double NumPy array
        source vectors, where each column corresponds to one source. Size (NNodes, Nsources).

    """
    
    link = copy.deepcopy(mesh.link)
    active = np.unique(link[link[:,2]==1,0]) - 1
    qvec = np.zeros((mesh.nodes.shape[0], active.size), dtype=np.complex128)
    if len(mesh.source.int_func) == 0:
        [ind, int_func] = utils.pointLocation(mesh, mesh.source.coord)
        print('int function calculated')
    else:
        ind = np.int32(mesh.source.int_func[:, 0]) - 1 # to zero-indexing
        int_func = mesh.source.int_func[:, 1:]
    
    dim = mesh.dimension
    qvec = sparse.csc_matrix((int_func[active,:].flatten(), (np.int32(mesh.elements[ind,:]-1).flatten(), np.repeat(np.arange(active.size),dim+1))),
                             shape=(mesh.nodes.shape[0], active.size), dtype=np.complex128)
    return qvec

def gen_sources_fl(mesh, phix, frequency=0., solver=utils.get_solver(), GPU=-1):
    """
    Calculates FEM sources vector for re-emission.
    
    If calculation fails on GPU (if chosen), it will generate a warning and automatically switch to CPU.

    Parameters
    ----------
    mesh : nirfasterff.base.fluormesh
        mesh used to calcualte the source vectors. Source information is also defined here.
    phix : double NumPy array
        excitation fluence calculated at each node for each source. Size (NNodes, NSources)
    frequency : double, optional
        modulation frequency, in Hz. The default is 0..
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    GPU : int, optional
        GPU selection. -1 for automatic, 0, 1, ... for manual selection on multi-GPU systems. The default is -1.

    Raises
    ------
    RuntimeError
        if both CUDA and CPU versions fail.
    TypeError
        if 'solver' is not 'CPU' or 'GPU'.

    Returns
    -------
    qvec : double or complex double NumPy array
        calculated fluence emission source vectors. Size (NNodes, NSources)

    """
    
    link = copy.deepcopy(mesh.link)
    active = np.unique(link[link[:,2]==1,0]) - 1
    nsource = active.size
    
    omega = 2. * np.pi * frequency
    gamma = (mesh.eta * mesh.muaf) / (1. + (omega * mesh.tau)**2)
    if frequency == 0:
        beta = gamma
        beta[beta==0] = 1e-20
    else:
        beta = gamma * (1 - (omega * mesh.tau * 1j))
        beta[beta==0] = 1e-20 + 1j*1e-20
        
    if solver.lower()=='gpu' and not utils.isCUDA():
        solver = 'CPU'
        print('Warning: No capable CUDA device found. using CPU instead')
    
    if solver.lower()=='gpu' and utils.isCUDA():
        try:
            qvec = utils.cudalib.gen_source_fl(mesh.nodes, mesh.elements, np.tile(np.atleast_2d(beta).T, nsource)*phix, GPU)
        except:
            print('Warning: GPU solver failed. Rolling back to CPU solver')
            try:
                qvec = utils.cpulib.gen_source_fl(mesh.nodes, mesh.elements, np.tile(np.atleast_2d(beta).T, nsource)*phix)
            except:
                raise RuntimeError('gen_source_fl failed')
    elif solver.lower()=='cpu':
        try:
            qvec = utils.cpulib.gen_source_fl(mesh.nodes, mesh.elements, np.tile(np.atleast_2d(beta).T, nsource)*phix)
        except:
            raise RuntimeError('gen_source_fl failed')
    else:
        raise TypeError('Solver should be ''GPU'' or ''CPU''')
    
    if frequency==0:
        qvec = np.ascontiguousarray(np.abs(qvec)) # should already be real, but just to be safe
    
    return qvec

def get_boundary_data(mesh, phi):
    """
    Calculates boundary data given the field data in mesh
    
    The field data can be any of the supported type: fluence, TPSF, or moments

    Parameters
    ----------
    mesh : nirfasterff mesh type
        the mesh whose boundary and detectors are used for the calculation.
    phi : double or complex double NumPy array
        field data as calculated by one of the 'get_field_*' solvers. Size (NNodes, NSources)

    Returns
    -------
    data : double or complex double NumPy array
        measured boundary data at each channel. Size (NChannels,).

    """
   
    if len(mesh.meas.int_func) == 0:
        print('Calculating missing detectors integration functions.')
        ind, int_func = utils.pointLocation(mesh, mesh.meas.coord)
    else:
        ind = np.int32(mesh.meas.int_func[:,0]) - 1
        int_func = mesh.meas.int_func[:,1:]
    
    link = copy.deepcopy(mesh.link)
    link[:,:2] -= 1  # to zero-indexing
    active_src = list(np.unique(link[link[:,2]==1,0]))
    bnd = mesh.bndvtx>0
    data = np.zeros(link.shape[0], dtype=phi.dtype)

    for i in range(link.shape[0]):
        if link[i,2]==0:
            data[i] = np.nan
            continue
        tri = list(np.int32(mesh.elements[ind[link[i,1]], :] - 1))
        int_func_tmp = int_func[link[i,1],:] * bnd[tri]
        int_func_tmp /= int_func_tmp.sum()
        data[i] = int_func_tmp.dot(phi[tri, active_src.index(link[i,0])])
    
    return data
