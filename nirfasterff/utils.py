"""
Utility functions and auxiliary classes frequently used in the package.
"""

import numpy as np
from scipy import sparse
from scipy import spatial
from scipy import integrate
from nirfasterff.lib import nirfasterff_cpu as cpulib
if cpulib.isCUDA():
    from nirfasterff.lib import nirfasterff_cuda as cudalib
import os
import psutil

class SolverOptions:
    """
    Parameters used by the FEM solvers, Equivalent to 'solver_options' in the Matlab version
    
    Attributes
    ----------
    max_iter: int
        maximum number of iterations allowed. Default: 1000
    AbsoluteTolerance: double
        Absolute tolerance for convergence. Default: 1e-12
    RelativeTolerance: double
        Relative (to the initial residual norm) tolerance for convergence. Default: 1e-12
    divergence: double
        Stop the solver when residual norm greater than this value. Default: 1e8
    GPU: int
        GPU selection. -1 for automatic, 0, 1, ... for manual selection on multi-GPU systems. Default: -1

    """
    def __init__(self, max_iter = 1000, AbsoluteTolerance = 1e-12, RelativeTolerance = 1e-12, divergence = 1e8, GPU = -1):
        self.max_iter = max_iter
        self.AbsoluteTolerance = AbsoluteTolerance
        self.RelativeTolerance = RelativeTolerance
        self.divergence = divergence
        self.GPU = GPU
        
class ConvergenceInfo:
    """
    Convergence information of the FEM solvers. Only used internally as a return type of functions nirfasterff.math.get_field_*
    
    Constructed using the output of the internal C++ functions

    Attributes
    ----------
        isConverged: bool array
            if solver converged to relative tolerance, for each rhs
        isConvergedToAbsoluteTolerance: bool array
            if solver converged to absolute tolerance, for each rhs
        iteration: int array
            iterations taken to converge, for each rhs
        residual: double array
            final residual, for each rhs

    """
    def __init__(self, info = None):
        self.isConverged = []
        self.isConvergedToAbsoluteTolerance = []
        self.iteration = []
        self.residual = []
        if info != None:
            for item in info:
                self.isConverged.append(item.isConverged)
                self.isConvergedToAbsoluteTolerance.append(item.isConvergedToAbsoluteTolerance)
                self.iteration.append(item.iteration)
                self.residual.append(item.residual)

class MeshingParams:
    """
    Parameters to be used by the CGAL mesher. Note: they should all be double

    Attributes
    ----------
        xPixelSpacing: double
            voxel distance in x direction. Default: 1.0
        yPixelSpacing: double
            voxel distance in y direction. Default: 1.0
        SliceThickness: double
            voxel distance in z direction. Default: 1.0
        facet_angle:double
            lower bound for the angle (in degrees) of surface facets. Default: 25.0
        facet_size:double
            upper bound for the radii of surface Delaunay balls circumscribing the facets. Default: 3.0
        facet_distance:double
            upper bound for the distance between the circumcenter of a surface facet and the center of its surface Delaunay ball. Default: 2.0
        cell_radius_edge:double
            upper bound for the ratio between the circumradius of a mesh tetrahedron and its shortest edge. Default: 3.0
        general_cell_size:double
            upper bound on the circumradii of the mesh tetrahedra, when no region-specific parameters (see below) are provided. Default: 3.0
        subdomain: double Numpy array
            Specify cell size for each region, in format::
                
                [region_label1, cell_size1]
                [region_label2, cell_size2]
                    ...
                                                
            If a region is not specified, value in "general_cell_size" will be used.  Default: np.array([0., 0.])
        lloyd_smooth: bool
            Switch for Lloyd smoother before local optimization. This can take up to 120s (hard limit set) but improves mesh quality. Default: True
        offset: double Numpy array
            offset value to be added to the nodes after meshing. Size (3,). Defualt: None
    
    Notes
    ----------
    Refer to CGAL documentation for details of the meshing algorithm as well as its parameters
    
    https://doc.cgal.org/latest/Mesh_3/index.html#Chapter_3D_Mesh_Generation

    """
    def __init__(self, xPixelSpacing=1., yPixelSpacing=1., SliceThickness=1.,
                             facet_angle = 25., facet_size = 3., facet_distance = 2.,
                             cell_radius_edge = 3., general_cell_size = 3., subdomain = np.array([0., 0.]),
                             lloyd_smooth = True, offset = None):
        self.xPixelSpacing = xPixelSpacing
        self.yPixelSpacing = yPixelSpacing
        self.SliceThickness = SliceThickness
        self.facet_angle = facet_angle
        self.facet_size = facet_size
        self.facet_distance = facet_distance
        self.cell_radius_edge = cell_radius_edge
        self.general_cell_size = general_cell_size
        self.subdomain = subdomain
        self.smooth = lloyd_smooth
        self.offset = offset

def isCUDA():
    """
    Checks if system has a CUDA device with compute capability >=5.2
    
    On a Mac machine, it automatically returns False without checking

    Returns
    -------
    bool
        True if a CUDA device with compute capability >=5.2 exists, False if not.

    """
    return cpulib.isCUDA()

def get_solver():
    """
    Get the default solver.

    Returns
    -------
    str
        If isCUDA is true, returns 'GPU', otherwise 'CPU'.

    """
    if isCUDA():
        solver = 'GPU'
    else:
        solver = 'CPU'     
    return solver            

def pointLocation(mesh, pointlist):
    """
    Similar to Matlab's pointLocation function, queries which elements in mesh the points belong to, and also calculate the barycentric coordinates.
    
    This is a wrapper of the C++ function pointLocation, which implememnts an AABB tree based on Darren Engwirda's findtria package

    Parameters
    ----------
    mesh : NIRFASTer mesh
        Can be any of the NIRFASTer mesh types (stnd, fluor, dcs). 2D or 3D.
    pointlist : NumPy array
        A list of points to query. Shape (N, dim), where N is number of points.

    Returns
    -------
    ind : double NumPy array
        i-th queried point is in element `ind[i`] of mesh (zero-based). If not in mesh, `ind[i]=-1`. Size: (N,).
    int_func : double NumPy array
        i-th row is the barycentric coordinates of i-th queried point. If not in mesh, corresponding row is all zero. Size: (N, dim+1).
        
    References
    -------
    https://github.com/dengwirda/find-tria

    """
    try:
        ind, int_func = cpulib.pointLocation(mesh.elements, mesh.nodes, np.atleast_2d(pointlist*1.0))
    except:
        print('Warning: pointLocation failed. Returning zero results', flush=1)
        ind = 0.0
        int_func = 0.0
    return ind, int_func

def check_element_orientation_2d(ele, nodes):
    """
    Make sure the 2D triangular elements are oriented counter clock wise.
    
    This is a direct translation from the Matlab version.

    Parameters
    ----------
    ele : NumPy array
        Elements in a 2D mesh. One-based. Size: (NNodes, 3).
    nodes : NumPy array
        Node locations in a 2D mesh. Size: (NNodes, 2).

    Raises
    ------
    TypeError
        If ele does not have three rows, i.e. not a 2D triangular mesh.

    Returns
    -------
    ele : NumPy array
        Re-oriented element list.

    """
    if ele.shape[1] != 3:
        raise TypeError('check_element_orientation_2d expects a 2D triangular mesh!')
    if nodes.shape[1] == 2:
        nodes = np.c_[nodes, np.zeros(nodes.shape[0])]
    v1 = nodes[np.int32(ele[:,1]-1),:] - nodes[np.int32(ele[:,0]-1),:]
    v2 = nodes[np.int32(ele[:,2]-1),:] - nodes[np.int32(ele[:,0]-1),:]
    
    z = np.cross(v1, v2)
    idx = z[:,2]<0
    if np.any(idx):
        ele[np.ix_(idx, [0,1])] = ele[np.ix_(idx, [1,0])]
    return ele

def pointLineDistance(A, B, p):
    """
    Calculate the distance between a point and a line (defined by two points), and find the projection point
    
    This is a direct translation  from the Matlab version

    Parameters
    ----------
    A : NumPy array
        first point on the line. Size (2,) or (3,)
    B : NumPy array
        second point on the line. Size (2,) or (3,)
    p : NumPy array
        point of query. Size (2,) or (3,)

    Returns
    -------
    dist : double
        point-line distance.
    point : NumPy array
        projection point on the line.

    """
    t = np.dot(p-A, B-A) / np.dot(B-A, B-A)
    if t<0:
        t = 0
    elif t>1:
        t = 1
    
    point = A + (B-A)*t
    dist = np.linalg.norm(p - point)
    return dist, point

def pointTriangleDistance(TRI, P):
    """
    Calculate the distance between a point and a triangle (defined by three points), and find the projection point

    Parameters
    ----------
    TRI : Numpy array
        The three points (per row) defining the triangle. Size: (3,3)
    P : Numpy array
        point of query. Size (3,).

    Returns
    -------
    dist : double
        point-triangle distance.
    PP0 : NumPy array
        projection point on the triangular face.
        
    Notes
    -----
    This is modified from Joshua Shaffer's code, available at: https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e
    
    which is based on Gwendolyn Fischer's Matlab code: https://uk.mathworks.com/matlabcentral/fileexchange/22857-distance-between-a-point-and-a-triangle-in-3d

    """
    
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = np.sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1
    return dist, PP0

def gen_intmat_impl(mesh, xgrid, ygrid, zgrid):
    """
    YOU SHOULD NOT USE THIS FUNCTION DIRECTLY. USE MESH.GEN_INTMAT INSTEAD.
    
    Heart of the gen_intmat function, which calculates the necessary information for converting between mesh and grid space

    Parameters
    ----------
    mesh : nirfasterff.base.stndmesh or nirfasterff.base.fluormesh
        The original mesh with with FEM data is calculated.
    xgrid : float64 NumPy 1-D array
        x grid in mm. Must be regular.
    ygrid : float64 NumPy 1-D array
        y grid in mm. Must be regular.
    zgrid : float64 NumPy 1-D array, or [] if 2D mesh
        z grid in mm. Must be regular.

    Returns
    -------
    gridinmesh: int32 NumPy array
        Col 0: indices of grid points that are in the mesh; Col 1: indeces of the elements the grid point is in. Flattened in 'F' order, one-based.
    meshingrid: int32 NumPy array
        Indices of mesh nodes that are in the grid. One-based.
    int_mat_mesh2grid : CSC sparse matrix, float64
        Sparse matrix converting data from mesh space to grid space. Size (NGrid, NNodes).
    int_mat_mesh2grid : CSC sparse matrix, float64
        Sparse matrix converting data from grid space to mesh space. Size (NNodes, NGrid).

    """
    if len(zgrid)==0:
        X, Y = np.meshgrid(xgrid, ygrid)
        coords = np.c_[X.flatten('F'), Y.flatten('F')]
    else:
        X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid)
        coords = np.c_[X.flatten('F'), Y.flatten('F'), Z.flatten('F')]
        
    ind, int_func = cpulib.pointLocation(mesh.elements, mesh.nodes, np.atleast_2d(coords*1.0))
    gridinmesh = np.flatnonzero(ind>-1) # This is zero-based
    int_func_inside = int_func[gridinmesh, :]

    nodes = np.int32(mesh.elements[ind[gridinmesh],:] - 1)
    int_mat_mesh2grid = sparse.csc_matrix((int_func_inside.flatten('F'), (np.tile(gridinmesh, int_func.shape[1]), nodes.flatten('F'))), shape=(ind.size, mesh.nodes.shape[0]))
    
    # Now calculate the transformation from grid to mesh
    # We can cheat a little bit because of the regular grid: we can triangularize one voxel and replicate
    if len(zgrid)>0:
        res = np.array([xgrid[1]-xgrid[0], ygrid[1]-ygrid[0], zgrid[1]-zgrid[0]])
    else:
        res = np.array([xgrid[1]-xgrid[0], ygrid[1]-ygrid[0]])
        
    if len(zgrid)>0:
        start = np.array([xgrid[0], ygrid[0], zgrid[0]])
        nodes0 = np.array([[0,0,0], [0, res[1], 0], 
                           [res[0], 0, 0], [res[0],res[1],0], 
                           [0,0,res[2]], [0, res[1], res[2]], 
                           [res[0], 0, res[2]], [res[0],res[1],res[2]]])
        # hard-coded element list
        ele0 = np.array([[2,5,6,3],
                         [5,7,6,3],
                         [3,2,5,1],
                         [1,2,5,4],
                         [0,2,1,4],
                         [5,2,6,4]], dtype=np.int32)
        # Calculate integration function within the small cube
        loweridx = np.floor((mesh.nodes - start) / res)
        pos_in_cube = mesh.nodes - (loweridx * res + start)
        ind0, int_func0 = cpulib.pointLocation(np.float64(ele0+1), 1.0*nodes0, pos_in_cube)
        # Convert back to the node numbering of the full grid
        raw_idx = np.zeros((mesh.nodes.shape[0], 4))
        for i in range(mesh.nodes.shape[0]):
            cube_coord = nodes0 + (loweridx[i,:] * res + start)
            tet_vtx = cube_coord[ele0[ind0[i], :], :]
            rel_idx = (tet_vtx - start) / res
            raw_idx[i,:] = rel_idx[:,2]*len(xgrid)*len(ygrid) + rel_idx[:,0]*len(ygrid) + rel_idx[:,1] # zero-based
        
        outvec = (loweridx[:,0]<0) | (loweridx[:,1]<0) | (loweridx[:,2]<0)
        meshingrid = np.flatnonzero(~outvec)
        # if any of the queried nodes was not asigned a value in the previous step,
        # treat it as an outside node and extrapolate. Otherwise the boundary elements will have smaller values than they should
        tmp = raw_idx[meshingrid, :]
        tmp2 = np.isin(tmp, gridinmesh)
        outside = np.r_[np.flatnonzero(outvec), meshingrid[tmp2.sum(axis=1)<tmp2.shape[1]]]
        meshingrid = np.array(list(set(meshingrid) - set(outside)))
    else:
        start = np.array([xgrid[0], ygrid[0]])
        nodes0 = np.array([[0,0], [0, res[1]], 
                           [res[0], 0], [res[0],res[1]]])
        # hard-coded element list
        ele0 = np.array([[2,1,0],
                         [1,2,3]], dtype=np.int32)
        # Calculate integration function within the small cube
        loweridx = np.floor((mesh.nodes - start) / res)
        pos_in_cube = mesh.nodes - (loweridx * res + start)
        ind0, int_func0 = cpulib.pointLocation(np.float64(ele0+1), 1.0*nodes0, pos_in_cube)
        # Convert back to the node numbering of the full grid
        raw_idx = np.zeros((mesh.nodes.shape[0], 3))
        for i in range(mesh.nodes.shape[0]):
            cube_coord = nodes0 + (loweridx[i,:] * res + start)
            tet_vtx = cube_coord[ele0[ind0[i], :], :]
            rel_idx = (tet_vtx - start) / res
            raw_idx[i,:] = rel_idx[:,0]*len(ygrid) + rel_idx[:,1] # zero-based
        
        outvec = (loweridx[:,0]<0) | (loweridx[:,1]<0)
        meshingrid = np.flatnonzero(~outvec)
        # if any of the queried nodes was not asigned a value in the previous step,
        # treat it as an outside node and extrapolate. Otherwise the boundary elements will have smaller values than they should
        tmp = raw_idx[meshingrid, :]
        tmp2 = np.isin(tmp, gridinmesh)
        outside = np.r_[np.flatnonzero(outvec), meshingrid[tmp2.sum(axis=1)<tmp2.shape[1]]]
        meshingrid = np.array(list(set(meshingrid) - set(outside)))
    
    gridTree = spatial.KDTree(coords[gridinmesh, :])
    _,nn = gridTree.query(mesh.nodes[outside,:])
    int_func_inside = int_func0[meshingrid, :]
    nodes = np.int64(raw_idx[meshingrid,:])
    int_mat_grid2mesh = sparse.csc_matrix((np.r_[int_func_inside.flatten('F'), np.ones(nn.size)], 
                                 (np.r_[np.tile(meshingrid, int_func.shape[1]), outside], np.r_[nodes.flatten('F'), gridinmesh[nn]])), shape=(ind0.size, coords.shape[0]))
    
    return np.c_[gridinmesh+1, ind[gridinmesh]], meshingrid+1, int_mat_mesh2grid, int_mat_grid2mesh # convert to one-based
    

def compress_coo(coo_idx, N):
    """
    Convert COO indices to compressed. 

    Parameters
    ----------
    coo_idx : int NumPy array
        Input indices in COO format, zero-based.
    N : int
        Number of rows in the sparse matrix.

    Returns
    -------
    int NumPy array
        Output indices in compressed format, zero-based. Size (N+1,)

    """
    cnt = np.zeros(N, dtype=np.int32)
    for i in range(coo_idx.size):
        cnt[coo_idx[i]] += 1
    return np.r_[0, np.cumsum(cnt)]

def uncompress_coo(compressed_idx):
    """
    Convert compressed indices to COO.

    Parameters
    ----------
    compressed_idx : int NumPy array
        Input indices in compressed format, zero-based.

    Returns
    -------
    coo_idx : int NumPy array
        Output indices in COO format, zero-based.

    """
    coo_idx = np.zeros(compressed_idx[-1], dtype=np.int32)
    for i in range(compressed_idx.size-1):
        for j in range(compressed_idx[i], compressed_idx[i+1]):
            coo_idx[j] = i
    return coo_idx


def boundary_attenuation(n_incidence, n_transmission=1.0, method='robin'):
    """
    Calculate the boundary attenuation factor between two media. 
    
    If vectors are used as inputs, they must have the same size and calculation is done for each pair
    
    If n_incidence is a vector but n_transmission is a scalar, code assumes n_transmission to be the same for each value in n_incidence

    Parameters
    ----------
    n_incidence : double Numpy vector or scalar
        refractive index of the medium within the boundary, e.g. a tissue.
    n_transmission : double Numpy vector or scalar, optional
        refractive index of the medium outside of the boundary, e.g. air. The default is 1.0.
    method : str, optional
        boundary type, which can be,
        
        'robin'  - internal reflectance derived from Fresnel's law
        
        'approx' - Groenhuis internal reflectance approximation :math:`(1.440n^{-2} + 0.710n^{-1} + 0.668 + 0.00636n)`
        
        'exact'  - exact internal reflectance (integrals of polarised reflectances, etc.) 
        
        The default is 'robin'.

    Raises
    ------
    ValueError
        if n_incidence and n_transmission are both vectors and have difference sizes, or if method is not of a recognized kind

    Returns
    -------
    A : double Numpy vector or scalar
        calculated boundary attenuation factor.
        
    References
    -------
    Durduran et al, 2010, Rep. Prog. Phys. doi:10.1088/0034-4885/73/7/076701

    """
    if np.size(n_incidence)>1 and np.size(n_transmission)==1:
        n_transmission = n_transmission*np.ones(np.size(n_incidence))
    elif np.size(n_incidence)>1 and np.size(n_incidence)!=np.size(n_transmission):
        raise ValueError('n_incidence and n_transmission size mismatch')
    n_incidence = np.atleast_1d(n_incidence)
    n_transmission = np.atleast_1d(n_transmission)
    def fresnel(angle, n_incid, n_transmit): 
        n = n_incid/n_transmit
        # S-polarized
        Rs = np.abs((n_incid*np.cos(angle) - n_transmit*np.emath.sqrt(1. - (n*np.sin(angle))**2)) / (n_incid*np.cos(angle) + n_transmit*np.emath.sqrt(1. - (n*np.sin(angle))**2)))**2
        # P-polarized
        Rp = np.abs((n_incid*np.emath.sqrt(1. - (n*np.sin(angle))**2) - n_transmit*np.cos(angle)) / (n_incid*np.emath.sqrt(1. - (n*np.sin(angle))**2) + n_transmit*np.cos(angle)))**2
        # return unpolarized
        return 0.5*(Rs+Rp)
    
    if method.lower() == 'robin':
        n = n_incidence/n_transmission
        R0 = (n-1)*(n-1) / ((n+1)*(n+1))
        # critical angle
        theta_incidence = np.arcsin(1/n)
        A = (2.0/(1-R0) - 1. + np.abs(np.cos(theta_incidence))**3) / (1. - np.abs(np.cos(theta_incidence))**2)    
    elif method.lower() == 'approx':
        n = n_incidence/n_transmission
        Reff = -1.44*n**(-2) + 0.71*n**(-1) + 0.668 + 0.0636*n
        A = (1. + Reff) / (1. - Reff)
    elif method.lower() == 'exact':
        A = np.zeros(np.size(n_incidence))
        # reflectance fluence rate at tissue-something boundary (from n_incidence to n_transmission)
        fun_fi = lambda x, ni, nt: np.sin(2.0*x) * fresnel(x, ni, nt)
        # reflectance current density at tissue-something boundary (from n_incidence to n_transmission)
        fun_j = lambda x, ni, nt: 3.*np.sin(x)*np.cos(x)*np.cos(x) * fresnel(x, ni, nt)
        for i in range(np.size(n_incidence)):
            Rfi,_ = integrate.quad(fun_fi, 0., 0.5*np.pi, args=(n_incidence[i], n_transmission[i]))
            Rj,_ = integrate.quad(fun_j, 0., 0.5*np.pi, args=(n_incidence[i], n_transmission[i]))
            # reflectance at boundary
            Reff = (Rfi + Rj) / (2.0 - Rfi + Rj)
            A[i] = (1. + Reff) / (1. - Reff)
    else:
        raise ValueError('Unknown boundary type')
    return A

def get_nthread():
    '''
    Choose the number of OpenMP threads in CPU solvers
    
    On CPUs with no hyperthreading, all physical cores are used
    Otherwise use min(physical_core, 8), i.e. no more than 8
    
    This is heuristically determined to avoid performance loss due to memory bottlenecking
    
    Advanced user can directly modify this function to choose the appropriate number of threads

    Returns
    -------
    nthread : int
        number of OpenMP threads to use in CPU solvers.

    '''
    logic_core = os.cpu_count()
    physical_core = psutil.cpu_count(0)
    if logic_core==physical_core:
        # no hyperthreading, use all physical cores
        nthread = physical_core
    else:
        # Use up to 8 threads to avoid memory bottleneck
        nthread = np.min((physical_core, 8))
    return nthread
        