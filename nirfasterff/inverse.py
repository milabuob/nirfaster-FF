"""
Calculation of the Jacobian matrices and a basic Tikhonov regularization function
"""
from nirfasterff import utils
from nirfasterff import base
import numpy as np
import copy
from scipy import sparse
from scipy import linalg

def tikhonov(A, reg, y):
    """
    Solves Tikhonov regularization (ie ridge regression)
    
    That is, given a linear system y = Ax, it solves :math:`\\arg\\min_x ||Ax-y||_2^2+||\\Gamma x||_2^2` 
    
    where A is the forward matrix, y is the recording, and :math:`\\Gamma` is the regularization matrix

    Parameters
    ----------
    A : double NumPy array
        forward matrix, e.g. the Jacobian in DOT.
    reg : double NumPy array or scalar
        if a scalar, the same regularization is applied to all elements of x, i.e. :math:`\\Gamma=reg*I`.
        
        if a vector, the regularization matrix is assumed to be diagonal, with the diagonal elements specified in reg, i.e. :math:`\\Gamma=diag(reg)`.
        
        if a matrix, it must be symmetric. In this case, :math:`\\Gamma=reg`.
    y : double NumPy vector
        measurement vector, e.g. dOD at each channel in DOT.

    Raises
    ------
    ValueError
        if reg is of incompatible size or not symmetric (if matrix).

    Returns
    -------
    result : double NumPy vector
        Tikhonov regularized solution to the linear system.

    """

    if np.isscalar(reg):
        reg = 1.0*reg
        reg_type = 0
    else:
        if len(reg)>1:
            if reg.ndim == 1:
                if len(reg)!=A.shape[1]:
                    raise ValueError('regularizer size mismatch')
                reg_type = 1
            else:
                if reg.shape[0] != reg.shape[1] or not linalg.issymmetric(reg):
                    raise ValueError('regularizer must be symmetric')
                elif reg.shape[0] != A.shape[1]:
                    raise ValueError('regularizer size mismatch')
                reg_type = 2
    
    m = A.shape[0]
    n = A.shape[1]
    if m >= n:
        if reg_type == 0:
            tmp = A.T @ A + reg*np.eye(n)
        elif reg_type == 1:
            tmp = A.T @ A + np.diag(reg)
        else:
            tmp = A.T @ A + reg.T @ reg
        beta = A.T @ y
        result = linalg.solve(tmp, beta, overwrite_b=1, assume_a='sym')
    else:
        if reg_type == 0 or reg_type == 1:
            if reg_type == 0:
                A2T = A.T / reg
            else:
                A2T = A.T / reg[:,None]
            beta = A2T @ y
            result = beta / reg
        else:
            L = linalg.cholesky(reg.T @ reg, lower=1)
            A2T = linalg.solve_triangular(L, A.T, lower=1)
            beta = A2T @ y
            result = linalg.solve_triangular(L.T, beta)
        beta = A2T.T @ beta
        tmp = A2T.T @ A2T
        tmp += np.eye(m)
        beta = linalg.solve(tmp, beta, overwrite_b=1, assume_a='sym')
        beta = A2T @ beta
        
        if reg_type == 0 or reg_type == 1:
            beta = beta / reg
        else:
            beta = linalg.solve_triangular(L.T, beta, overwrite_b=1)
        result = result - beta
        
    return result

def jacobian_stnd_FD(mesh, freq, normalize=True, mus=True, solver = utils.get_solver(), opt = utils.SolverOptions()):
    """
    Calculates the frequency-domain Jacobian matrix using the adjoint method
    
    Calculates spatial distributions of sensitivity of field registerd on the mesh boundary to changes of optical properties per voxel.
    
    When freq=0, see :func:`~nirfasterff.inverse.jacobian_stnd_CW()` for the structure of the Jacobian
    
    When freq>0, the Jacobian is structured as, suppose we have M channels and N voxels::
        
        [dA_1/dmusp_1, dA_1/dmusp_2, ..., dA_1/dmusp_{N}, dA_1/dmua_1, dA_1/dmua_2, ..., dA_1/dmua_{N}]
        [dPhi_1/dmusp_1, dPhi_1/dmusp_2, ..., dPhi_1/dmusp_{N}, dPhi_1/dmua_1, dPhi_1/dmua_2, ..., dPhi_1/dmua_{N}]
        [dA_2/dmusp_1, dA_2/dmusp_2, ..., dA_2/dmusp_{N}, dA_2/dmua_1, dA_2/dmua_2, ..., dA_2/dmua_{N}]
        [dPhi_2/dmusp_1, dPhi_2/dmusp_2, ..., dPhi_2/dmusp_{N}, dPhi_2/dmua_1, dPhi_2/dmua_2, ..., dPhi_2/dmua_{N}]
        ...
        [dA_M/dmusp_1, dA_M/dmusp_2, ..., dA_M/dmusp_{N}, dA_M/dmua_1, dA_M/dmua_2, ..., dA_M/dmua_{N}]
        [dPhi_M/dmusp_1, dPhi_M/dmusp_2, ..., dPhi_M/dmusp_{N}, dPhi_M/dmua_1, dPhi_M/dmua_2, ..., dPhi_M/dmua_{N}]
    
    where A and Phi denote the measured amplitude and the phase if `normalize=False`, and the log of them if `normalize=True`
        
    Note that the calculation is only done in the volumetric space

    Parameters
    ----------
    mesh : nirfasterff.base.stndmesh
        mesh on which the Jacobian is calculated.
    freq : double
        modulation frequency, in Hz.
    normalize : bool, optional
        whether normalize the Jacobian to the amplitudes of boundary measurements, i.e. use Rytov approximation. 
        
        The default is True.
    mus : bool, optional
        whether derivates wrt mus (left half of the 'full' Jacobian) is calculated. Only has effect when freq=0. The default is True.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if mesh is not a stnd mesh, or mesh.vol is not defined.

    Returns
    -------
    J : double or complex double NumPy array
        The Jacobian matrix. Size (NChannel*2, NVoxel*2) if freq>0, (NChannel, NVoxel*2) if freq=0 and mus=True, (NChannel, NVoxel) if freq=0 and mus=False
    data1 : nirfasterff.base.FDdata
        The calculated direct field. The same as directly calling mesh.femdata(freq)
    data2 : nirfasterff.base.FDdata
        The calculated adjoint field. The same as calling mesh.femdata(freq) AFTER swapping the locations of sources and detectors
        
    References
    ----------
    Arridge, Applied Optics, 1995. doi:10.1364/AO.34.007395

    """

    if not mesh.type=='stnd':
        raise TypeError('Must be a standard mesh')
    if not mesh.isvol():
        raise TypeError('Starting from this version of NIRFASTer, Jacobian calculation is only supported in grid space. Please run mesh.gen_intmat() first')
        
    # Let's make a duplicate of the mesh, so we don't accidentcally mess up the original
    mesh2 = base.stndmesh()
    mesh2.from_copy(mesh)
    mesh2.vol = base.meshvol()  # empty the vol field for now. This avoids automatic space conversion, and makes full jacobian calculation easier
    # Calculate direct field
    print('Calculating direct field...', flush=1)
    data1,_ = mesh2.femdata(freq, solver, opt)
    # swap sources and detectors
    src2 = copy.deepcopy(mesh2.meas)
    det2 = copy.deepcopy(mesh2.source)
    mesh2.source = src2
    mesh2.meas = det2
    mesh2.link = mesh.link[:, [1,0,2]]
    # calculate adjoint field
    print('Calculating adjoint field...', flush=1)
    data2,_ = mesh2.femdata(freq, solver, opt)
    
    # get the active indices
    active_idx = np.nonzero(mesh.link[:,2] == 1)[0]
    link = np.ascontiguousarray(np.atleast_2d(mesh.link[active_idx,:]))
    # Integrate
    print('Integrating...', flush=1)
    phi = np.ascontiguousarray((mesh.vol.mesh2grid @ data1.phi).T)
    aphi = np.ascontiguousarray((mesh.vol.mesh2grid @ data2.phi).T)
    if freq==0 and not mus:
        J = np.empty((mesh.link.shape[0], phi.shape[1]))
        J.fill(np.nan)
        tmp = -utils.cpulib.IntGrid(phi, aphi, link)*np.prod(mesh.vol.res)
        
        if not normalize:
            J[active_idx, :] = tmp
        else:
            J[active_idx, :] = tmp / data1.amplitude[active_idx,None]
    else:
        dim = mesh.dimension
        intfunc_grad = utils.cpulib.gradientIntfunc2(mesh2.elements, mesh2.nodes, np.ascontiguousarray(mesh.vol.gridinmesh[:,1], dtype=np.int32))
        nodes = np.int32(mesh.elements[mesh.vol.gridinmesh[:,1],:] - 1)
        gradx_mesh2grid = sparse.csc_matrix((intfunc_grad[:,0:dim+1].flatten('F'), (np.tile(mesh.vol.gridinmesh[:,0]-1, dim+1), nodes.flatten('F'))), shape=mesh.vol.mesh2grid.shape)
        grady_mesh2grid = sparse.csc_matrix((intfunc_grad[:,dim+1:2*(dim+1)].flatten('F'), (np.tile(mesh.vol.gridinmesh[:,0]-1, dim+1), nodes.flatten('F'))), shape=mesh.vol.mesh2grid.shape)
        dx_phi = (gradx_mesh2grid @ data1.phi).T
        dy_phi = (grady_mesh2grid @ data1.phi).T
        dx_aphi = (gradx_mesh2grid @ data2.phi).T
        dy_aphi = (grady_mesh2grid @ data2.phi).T
        kappa = mesh.vol.mesh2grid @ mesh.kappa
        if dim==2:
            tmp1 = 3*utils.cpulib.IntGradGrid(np.ascontiguousarray(dx_phi), np.ascontiguousarray(dy_phi), np.array([], dtype=dx_phi.dtype), 
                                             np.ascontiguousarray(dx_aphi), np.ascontiguousarray(dy_aphi), np.array([], dtype=dx_aphi.dtype), link, dim)*(kappa**2)*np.prod(mesh.vol.res)
        elif dim==3:
            gradz_mesh2grid = sparse.csc_matrix((intfunc_grad[:,2*(dim+1):3*(dim+1)].flatten('F'), (np.tile(mesh.vol.gridinmesh[:,0]-1, dim+1), nodes.flatten('F'))), shape=mesh.vol.mesh2grid.shape)
            dz_phi = (gradz_mesh2grid @ data1.phi).T
            dz_aphi = (gradz_mesh2grid @ data2.phi).T
            tmp1 = 3*utils.cpulib.IntGradGrid(np.ascontiguousarray(dx_phi), np.ascontiguousarray(dy_phi), np.ascontiguousarray(dz_phi), 
                                             np.ascontiguousarray(dx_aphi), np.ascontiguousarray(dy_aphi), np.ascontiguousarray(dz_aphi), link, dim)*(kappa**2)*np.prod(mesh.vol.res)
        
        tmp2 = -utils.cpulib.IntGrid(phi, aphi, link)*np.prod(mesh.vol.res)
        
        if not normalize:
            J = np.empty((mesh.link.shape[0], phi.shape[1]*2), dtype=data1.phi.dtype)
            J.fill(np.nan)
            J[active_idx, :] = np.c_[tmp1, tmp2]
        else:
            if freq==0:
                J = np.empty((mesh.link.shape[0], phi.shape[1]*2))
                J.fill(np.nan)
                J[active_idx, :] = np.c_[tmp1, tmp2] / data1.amplitude[active_idx,None]
            else:
                J = np.empty((mesh.link.shape[0]*2, phi.shape[1]*2))
                J.fill(np.nan)
                tmp = np.c_[tmp1, tmp2] / data1.complex[active_idx,None]
                J[active_idx*2, :] = np.real(tmp)
                J[active_idx*2+1, :] = np.imag(tmp)
    data1.togrid(mesh)
    data2.togrid(mesh)
    return J, data1, data2

def jacobian_stnd_CW(mesh, normalize=True, mus=False, solver = utils.get_solver(), opt = utils.SolverOptions()):
    """
    Calculates the continuous-wave Jacobian matrix using the adjoint method
    
    Calculates spatial distributions of sensitivity of field registerd on the mesh boundary to changes of optical properties per voxel.
        
    When mus is set to True, the Jacobian is structured as, suppose we have M channels and N voxels::
        
        [dA_1/dmusp_1, dA_1/dmusp_2, ..., dA_1/dmusp_{N}, dA_1/dmua_1, dA_1/dmua_2, ..., dA_1/dmua_{N}]
        [dA_2/dmusp_1, dA_2/dmusp_2, ..., dA_2/dmusp_{N}, dA_2/dmua_1, dA_2/dmua_2, ..., dA_2/dmua_{N}]
        ...
        [dA_M/dmusp_1, dA_M/dmusp_2, ..., dA_M/dmusp_{N}, dA_M/dmua_1, dA_M/dmua_2, ..., dA_M/dmua_{N}]
    
    where A and Phi denote the measured amplitude and the phase if `normalize=False`, and the log of them if `normalize=True`
    
    When mus is set to False, the returned Jacobian is only the right half of the above. That is, only derivatives wrt mua
    
    Note that the calculation is only done in the volumetric space

    Parameters
    ----------
    mesh : nirfasterff.base.stndmesh
        mesh on which the Jacobian is calculated.
    normalize : bool, optional
        whether normalize the Jacobian to the amplitudes of boundary measurements, i.e. use Rytov approximation. 
        
        The default is True.
    mus : bool, optional
        whether derivates wrt mus (left half of the 'full' Jacobian) is calculated. The default is False.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if mesh is not a stnd mesh, or mesh.vol is not defined.

    Returns
    -------
    J : double NumPy array
        The Jacobian matrix. Size (NChannel, NVoxel*2) if mus=True, (NChannel, NVoxel) if mus=False
    data1 : nirfasterff.base.FDdata
        The calculated direct field. The same as directly calling mesh.femdata(0)
    data2 : nirfasterff.base.FDdata
        The calculated adjoint field. The same as calling mesh.femdata(0) AFTER swapping the locations of sources and detectors

    """
    return jacobian_stnd_FD(mesh, 0., normalize, mus, solver, opt)

def jacobian_fl_CW(mesh, normalize=True, solver = utils.get_solver(), opt = utils.SolverOptions()):
    """
    Calculates the continuous-wave fluorescence Jacobian matrix using the adjoint method
    
    J_{ij} = dA_i / d_gamma_j
    
    where A is fluorescence amplitude if normalization is False, fluorescence amplitude divided by excitation amplitude ('Born ratio') if True
    
    gamma_j = mesh.eta[j]*mesh.muaf[j]

    Parameters
    ----------
    mesh : nirfasterff.base.fluormesh
        mesh on which the Jacobian is calculated.
    normalize : bool, optional
        whether normalize the Jacobian to the amplitudes of boundary measurements at excitation wavelength ('Born ratio'). 
        
        The default is True.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if mesh is not a fluor mesh, or mesh.vol is not defined.

    Returns
    -------
    J : double NumPy array
        The Jacobian matrix. Size (NChannel, NVoxel) 
    data1 : nirfasterff.base.FLdata
        The calculated direct field. The same as directly calling mesh.femdata(0)
    data2 : nirfasterff.base.FLdata
        The calculated adjoint field. The same as calling mesh.femdata(0) AFTER swapping the locations of sources and detectors
    
    References
    ----------
    Milstein et al., JOSA A, 2004. doi:10.1364/JOSAA.21.001035

    """
    if not mesh.type=='fluor':
        raise TypeError('Must be a fluorescence mesh')
    if not mesh.isvol():
        raise TypeError('Starting from this version of NIRFASTer, Jacobian calculation is only supported in grid space. Please run mesh.gen_intmat() first')
    # Let's make a duplicate of the mesh, so we don't accidentcally mess up the original
    mesh2 = base.fluormesh()
    mesh2.from_copy(mesh)
    # Calculate direct field
    print('Calculating direct field...', flush=1)
    data1 = mesh2.femdata(0, solver, opt, mmflag=False)[0]
    # swap sources and detectors
    src2 = copy.deepcopy(mesh2.meas)
    det2 = copy.deepcopy(mesh2.source)
    mesh2.source = src2
    mesh2.meas = det2
    mesh2.link = mesh.link[:, [1,0,2]]
    # calculate adjoint field
    print('Calculating adjoint field...', flush=1)
    data2 = mesh2.femdata(0, solver, opt, xflag=False, flflag=False)[0]
    
    # get the active indices
    active_idx = mesh.link[:,2] == 1
    link = np.ascontiguousarray(np.atleast_2d(mesh.link[active_idx,:]))
    # Integrate
    print('Integrating...', flush=1)
    phi = np.ascontiguousarray(np.reshape(data1.phix, (-1, data1.phix.shape[-1]), order='F').T)
    aphi = np.ascontiguousarray(np.reshape(data2.phimm, (-1, data2.phimm.shape[-1]), order='F').T)
    J = np.empty((mesh.link.shape[0], phi.shape[1]))
    J.fill(np.nan)
    tmp = utils.cpulib.IntGrid(phi, aphi, link)*np.prod(mesh.vol.res)
    
    if not normalize:
        J[active_idx, :] = tmp
    else:
        J[active_idx, :] = tmp / data1.amplitudex[active_idx,None]
        
    return J, data1, data2

def jacobian_fl_FD(mesh, freq, normalize=True, solver = utils.get_solver(), opt = utils.SolverOptions()):
    """
    Calculates the frequency-domain fluorescence Jacobian matrix using the adjoint method
    
    The Jacobian is structured as, suppose we have M channels and N voxels::
        
        [d_real{A_1}/d_gamma_1, d_real{A_1}/d_gamma_2, ..., d_real{A_1}/d_gamma_{N}, d_real{A_1}/dtau_1, d_real{A_1}/dtau_2, ..., d_real{A_1}/dtau_{N}]
        [d_imag{A_1}/d_gamma_1, d_imag{A_1}/d_gamma_2, ..., d_imag{A_1}/d_gamma_{N}, d_imag{A_1}/dtau_1, d_imag{A_1}/dtau_2, ..., d_imag{A_1}/dtau_{N}]
        [d_real{A_2}/d_gamma_1, d_real{A_2}/d_gamma_2, ..., d_real{A_2}/d_gamma_{N}, d_real{A_2}/dtau_1, d_real{A_2}/dtau_2, ..., d_real{A_2}/dtau_{N}]
        [d_imag{A_2}/d_gamma_1, d_imag{A_2}/d_gamma_2, ..., d_imag{A_2}/d_gamma_{N}, d_imag{A_2}/dtau_1, d_imag{A_2}/dtau_2, ..., d_imag{A_2}/dtau_{N}]
        ...
        [d_real{A_M}/d_gamma_1, d_real{A_M}/d_gamma_2, ..., d_real{A_M}/d_gamma_{N}, d_real{A_M}/dtau_1, d_real{A_M}/dtau_2, ..., d_real{A_M}/dtau_{N}]
        [d_imag{A_M}/d_gamma_1, d_imag{A_M}/d_gamma_2, ..., d_imag{A_M}/d_gamma_{N}, d_imag{A_M}/dtau_1, d_imag{A_M}/dtau_2, ..., d_imag{A_M}/dtau_{N}]
    
    where A is fluorescence amplitude if normalization is False, fluorescence amplitude divided by excitation amplitude ('Born ratio') if True
    gamma_j = mesh.eta[j]*mesh.muaf[j]

    Parameters
    ----------
    mesh : nirfasterff.base.fluormesh
        mesh on which the Jacobian is calculated.
    freq : double
        modulation frequency in Hz.
    normalize : bool, optional
        whether normalize the Jacobian to the amplitudes of boundary measurements at excitation wavelength ('Born ratio'). 
        
        The default is True.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if mesh is not a fluor mesh, or mesh.vol is not defined.

    Returns
    -------
    J : double NumPy array
        The Jacobian matrix. Size (NChannel*2, NVoxel*2) 
    data1 : nirfasterff.base.FLdata
        The calculated direct field. The same as directly calling mesh.femdata(freq)
    data2 : nirfasterff.base.FLdata
        The calculated adjoint field. The same as calling mesh.femdata(freq) AFTER swapping the locations of sources and detectors
    
    References
    ----------
    Milstein et al., JOSA A, 2004. doi:10.1364/JOSAA.21.001035

    """
    if not mesh.type=='fluor':
        raise TypeError('Must be a fluorescence mesh')
    if not mesh.isvol():
        raise TypeError('Starting from this version of NIRFASTer, Jacobian calculation is only supported in grid space. Please run mesh.gen_intmat() first')
    
    if freq==0:
        return jacobian_fl_CW(mesh, normalize, solver, opt)
    # Let's make a duplicate of the mesh, so we don't accidentcally mess up the original
    mesh2 = base.fluormesh()
    mesh2.from_copy(mesh)
    mesh2.vol = base.meshvol()  # empty the vol field for now. This avoids automatic space conversion, and makes full jacobian calculation easier
    # Calculate direct field
    print('Calculating direct field...', flush=1)
    data1 = mesh2.femdata(freq, solver, opt, mmflag=False)[0]
    # swap sources and detectors
    src2 = copy.deepcopy(mesh2.meas)
    det2 = copy.deepcopy(mesh2.source)
    mesh2.source = src2
    mesh2.meas = det2
    mesh2.link = mesh.link[:, [1,0,2]]
    # calculate adjoint field
    print('Calculating adjoint field...', flush=1)
    data2 = mesh2.femdata(freq, solver, opt, xflag=False, flflag=False)[0]
    # define gamma
    omega = 2*np.pi*freq
    # get the active indices
    active_idx = np.nonzero(mesh.link[:,2] == 1)[0]
    link = np.ascontiguousarray(np.atleast_2d(mesh.link[active_idx,:]))
    # Integrate
    print('Integrating...', flush=1)
    phi = np.ascontiguousarray((mesh.vol.mesh2grid @ data1.phix).T)
    aphi = np.ascontiguousarray((mesh.vol.mesh2grid @ data2.phimm).T)
    J = np.empty((mesh.link.shape[0]*2, phi.shape[1]*2))
    J.fill(np.nan)
    # The complex jacobian, wrt eta*muaf/(1+j*omega*tau)
    tmp = utils.cpulib.IntGrid(phi, aphi, link)*np.prod(mesh.vol.res)
    tau = mesh.vol.mesh2grid @ mesh.tau
    eta_muaf = mesh.vol.mesh2grid @ (mesh.eta * mesh.muaf)
    # Jacobian for eta*muaf
    tmp1 = tmp / (1.0 + 1j * omega * tau)
    # Jacobian for tau
    tmp2 = tmp * ((-1j * omega * eta_muaf) / (1.0 + 1j * omega * tau)**2)
    
    if not normalize:
        J[active_idx*2, :] = np.c_[np.real(tmp1), np.real(tmp2)]
        J[active_idx*2+1, :] = np.c_[np.imag(tmp1), np.imag(tmp2)]
    else:
        J[active_idx*2, :] = np.c_[np.real(tmp1/ data1.complexx[active_idx,None]), np.real(tmp2/ data1.complexx[active_idx,None])]
        J[active_idx*2+1, :] = np.c_[np.imag(tmp1/ data1.complexx[active_idx,None]), np.imag(tmp2/ data1.complexx[active_idx,None])]
        
    return J, data1, data2

def jacobian_DCS(mesh, tvec, normalize=True, solver = utils.get_solver(), opt = utils.SolverOptions()):
    """
    Calculates the Jacobian matrix for a DCS mesh using the adjoint method
    
    One Jacobian is calcualted at each time point in tvec, and the derivative is taken with regard to aDb

    Parameters
    ----------
    mesh : nirfasterff.base.dcsmesh
        mesh on which the Jacobian is calculated.
    tvec : double NumPy vector
        time vector used.
    normalize : bool, optional
        if True, Jacobbians are normalized to the measured boundary amplitude. The default is True.
    solver : str, optional
        Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
    opt : nirfasterff.utils.SolverOptions, optional
        Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
        
        See :func:`~nirfasterff.utils.SolverOptions` for details

    Raises
    ------
    TypeError
        if mesh is not a DCS mesh, or mesh.vol is not defined.

    Returns
    -------
    J : double NumPy array
        The Jacobian matrix. Size (NChannel, NVoxel, NTime) 
    data1 : nirfasterff.base.FLdata
        The calculated direct field. The same as directly calling mesh.femdata(tvec)
    data2 : nirfasterff.base.FLdata
        The calculated adjoint field. The same as calling mesh.femdata(tvec) AFTER swapping the locations of sources and detectors

    """
    if not mesh.type=='dcs':
        raise TypeError('Must be a DCS mesh')
    if not mesh.isvol():
        raise TypeError('Starting from this version of NIRFASTer, Jacobian calculation is only supported in grid space. Please run mesh.gen_intmat() first')
    # Let's make a duplicate of the mesh, so we don't accidentcally mess up the original
    mesh2 = base.dcsmesh()
    mesh2.from_copy(mesh)
    # Calculate direct field
    print('Calculating direct field...', flush=1)
    data1,_ = mesh2.femdata(tvec, solver, opt)
    # swap sources and detectors
    src2 = copy.deepcopy(mesh2.meas)
    det2 = copy.deepcopy(mesh2.source)
    mesh2.source = src2
    mesh2.meas = det2
    mesh2.link = mesh.link[:, [1,0,2]]
    # calculate adjoint field
    print('Calculating adjoint field...', flush=1)
    data2,_ = mesh2.femdata(tvec, solver, opt)
    # get the active indices
    active_idx = mesh.link[:,2] == 1
    link = np.ascontiguousarray(np.atleast_2d(mesh.link[active_idx,:]))
    # Integrate
    print('Integrating...', flush=1)
    J = np.empty((mesh.link.shape[0], np.prod(data1.phi_DCS.shape[:-2]), len(tvec)))
    J.fill(np.nan)
    k0 = 2*np.pi / (mesh.wv_DCS/1e6)
    phi = np.reshape(data1.phi_DCS, (-1, data1.phi_DCS.shape[-2], data1.phi_DCS.shape[-1]), order='F')
    aphi = np.reshape(data2.phi_DCS, (-1, data2.phi_DCS.shape[-2], data2.phi_DCS.shape[-1]), order='F')
    for i in range(len(tvec)):
        tmp_phi = np.ascontiguousarray(phi[:,:,i].T)
        tmp_aphi = np.ascontiguousarray(aphi[:,:,i].T)
        tmpJ = -utils.cpulib.IntGrid(tmp_phi, tmp_aphi, link)*np.prod(mesh.vol.res)
        musp = mesh.vol.mesh2grid @ mesh.mus
        if normalize:
            J[active_idx, :, i] = tmpJ*2*k0*k0*musp*tvec[i] / np.atleast_2d(data1.G1_DCS[:,i]).T
        else:
            J[active_idx, :, i] = tmpJ*2*k0*k0*musp*tvec[i]
    return J, data1, data2
