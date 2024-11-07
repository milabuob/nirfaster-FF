"""
Functions used for mesh generation and mesh quality check
"""
import numpy as np
from nirfasterff import utils
from nirfasterff import io
from . import auxiliary
# from lib import nirfasteruff_cpu
import os

def RunCGALMeshGenerator(mask, opt = utils.MeshingParams()):
    """
    Generate a tetrahedral mesh from a volume using CGAL 5.3.2 mesher, where different regions are labeled used a distinct integer.
    
    Also runs a pruning steps after the mesh generation, where nodes not referred to in the element list are removed.

    Parameters
    ----------
    mask : uint8 NumPy array
        3D volumetric data defining the space to mesh. Regions defined by different integers. 0 is background.
    opt : nirfasterff.utils.MeshingParams, optional
        meshing parameters used. Default values will be used if not specified.
        
        See :func:`nirfasterff.utils.MeshingParams` for details

    Returns
    -------
    ele : int NumPy array
        element list calculated by the mesher, one-based. Last column indicates the region each element belongs to
    nodes : double NumPy array
        element list calculated by the mesher, in mm.
    
    References
    -------
    https://doc.cgal.org/5.3.2/Mesh_3/index.html#Chapter_3D_Mesh_Generation,

    """
   
    if mask.dtype != 'uint8':
        print('Warning: CGAL only supports uint8. I am doing the conversion now, but this can lead to unexpected errors!', flush=1)
        mask = np.uint8(mask)
    
    tmpmeshfn = '._out.mesh'
    tmpinrfn = '._cgal_mesh.inr'

    # Save the tmp INRIA file
    io.saveinr(mask, tmpinrfn)
    
    # call the mesher
    utils.cpulib.cgal_mesher(tmpinrfn, tmpmeshfn, facet_angle=opt.facet_angle, facet_size=opt.facet_size,
                                 facet_distance=opt.facet_distance, cell_radius_edge_ratio=opt.cell_radius_edge,
                                 general_cell_size=opt.general_cell_size,subdomain=opt.subdomain, smooth=opt.smooth)
    # read result and cleanup
    ele_raw, nodes_raw, _, _ = io.readMEDIT(tmpmeshfn)
    if np.all(opt.offset != None):
        nodes_raw = nodes_raw + opt.offset
    
    ele_tmp = ele_raw[:,:-1]
    nids, ele = np.unique(ele_tmp, return_inverse=1)
    ele += 1 # to one-based
    nodes = nodes_raw[nids-1,:]
    ele = np.c_[ele.reshape((-1,4)), ele_raw[:,-1]]
    if nodes.shape[0] != nodes_raw.shape[0]:
        print(' Removed %d unused nodes from mesh!\n' % (nodes_raw.shape[0]-nodes.shape[0]), flush=1)
    # remove the tmpfiles
    os.remove(tmpmeshfn)
    os.remove(tmpinrfn)
    return ele, nodes

def boundfaces(nodes, elements, base=0, renumber=True):
    """
    Finds the boundary faces of a 3D tetrahedral mesh

    Parameters
    ----------
    nodes : double NumPy array
        node locations of the mesh. Size (NNodes, 3)
    elements : NumPy array
        element list of the mesh, can be either one-based or zero-based, which must be specified using the 'base' argument.
    base : int, optional
        one- or zero-based indexing of the element list. Can be 1, or 0. The default is 0.
    renumber : bool, optional
        whether renumber of the node indices in the extracted surface mesh. The default is True.

    Returns
    -------
    new_faces : int32 NumPy array
        list of boundary faces of the mesh. Base of indexing is consistent with input element list
        
        If `renumber=True`, node indices are renumbered; if not, same node indices as in 'elements' are used
    new_points : double NumPy array
        point locations of the boundary nodes.
        
        If `renumber=True`, returns the subset of node loations that are on the surface; if not, it is the same as input `nodes`

    """
    if base==1:
        ele = np.int32(elements-1)
    else:
        ele = np.int32(elements)
    faces = np.r_[ele[:, [0,1,2]], 
                  ele[:, [0,1,3]],
                  ele[:, [0,2,3]],
                  ele[:, [1,2,3]]]
    
    faces = np.sort(faces)
    unique_faces, cnt = np.unique(faces, axis=0, return_counts=1)
    ele_surf = unique_faces[cnt==1, :]
    node_idx = np.unique(ele_surf)
    if renumber:
        new_faces = np.searchsorted(node_idx, ele_surf)
        new_points = nodes[node_idx,:]
        if base==1:
            new_faces += 1
    else:
        new_faces = ele_surf
        new_points = nodes
    return new_faces, new_points

def checkmesh3d_solid(ele, nodes, verbose=False):
    """
    Calculates and returns the quality metrics of the tetrahedrons in a 3D tetrahedral mesh
    
    Please consider using :func:`nirfasterff.meshing.CheckMesh3D()` instead.

    Parameters
    ----------
    ele : int32 NumPy array
        element list of the mesh, zero-based.
    nodes : double NumPy array
        node locations of the mesh, in mm.
    verbose : bool, optional
        whether print the problematic tetrahedrons to stdout, if any. The default is False.

    Raises
    ------
    TypeError
        if mesh is not 3D tetrahedral, or if element list uses undefined nodes.

    Returns
    -------
    vol : double NumPy vector
        volume of each tetrahedron, mm^3.
    vol_ratio : double NumPy vector
        volume ratio, defined as the smallest sine of dihedral angles of each tetrahedron.
    zeroflag : bool NumPy vector
        flags of whether the volume of a tetrahedron is too small.
    faceflag : bool
        whether there are faces shared by more than two tetrahedrons.

    """
    # ele is zero-based, int
    if nodes.shape[1] != 3:
        raise TypeError('Mesh must be 3D')
    if (ele.shape[1] != 4) and (ele.shape[1] != 5):
        raise TypeError('Mesh must be a tetrahedral mesh')
        
    TetrahedronFailQuality = 0.03
    nnodes = nodes.shape[0]
    nodenumbers = np.arange(nnodes)
    
    # check for nodes not used by any tetrahedron
    tmp = np.isin(nodenumbers, ele)
    if not np.all(tmp):
        print('checkmesh3d_solid: The provided mesh has extra nodes that are not used in the element list', flush=1)
        print('Warning: Not all nodes are used in the element connectivity list', flush=1)
        if verbose:
            print(np.nonzero(~tmp)[0])
    
    # check for non-existing nodes used by tetrahedrons
    tmp = np.isin(ele, nodenumbers)
    if not np.all(tmp):
        print('checkmesh3d_solid: Some of the tets are using nodes that are not defined in node list!', flush=1)
        if verbose:
            print(np.nonzero(np.sum(tmp,axis=1)<4)[0], flush=1)
        raise TypeError('Some of the tets are using nodes that are not defined in node list!')
    
    # check for tetrahedrons with very small volume
    span = np.max(nodes, axis=0) - np.min(nodes, axis=0)
    tiny = span.max()*1e-6
    TetrahedronZeroVol = tiny
    vol = utils.cpulib.ele_area(nodes, np.float64(ele+1))
    print('Avg Min Max volume: %f %f %f\n' % (np.mean(vol), np.min(vol), np.max(vol)), flush=1)
    zeroflag = vol<=TetrahedronZeroVol
    
    # check for small volume ratio
    vol_ratio = auxiliary.simpqual(nodes, ele)
    qualflag = vol_ratio<=TetrahedronFailQuality
    print('Avg Min Max volume ratio quality: %f %f %f\n' % (np.mean(vol_ratio), np.min(vol_ratio), np.max(vol_ratio)), flush=1)
    nvoids = np.sum(qualflag)
    if nvoids:
        print('There are %d elements with undesirable quality.' % nvoids, flush=1)
        if verbose:
            print(np.nonzero(qualflag)[0], flush=1)
    
    # check if any faces are shared by more than two tetrahedrons
    faceflag = auxiliary.check_tetrahedron_faces(ele)
    
    return vol, vol_ratio, zeroflag, faceflag

def checkmesh3d_surface(ele, nodes, verbose=False):
    """
    Calculates and returns the quality metrics of the triangles in a 3D surface mesh
    
    Please consider using :func:`nirfasterff.meshing.CheckMesh3D()` instead.

    Parameters
    ----------
    ele : int32 NumPy array
        element list of the mesh, zero-based.
    nodes : double NumPy array
        node locations of the mesh, in mm.
    verbose : bool, optional
        whether print the problematic triangles to stdout, if any. The default is False.

    Raises
    ------
    TypeError
        if mesh is not 3D tetrahedral, or if element list uses undefined nodes.

    Returns
    -------
    q_radius_ratio : double NumPy vector
        radius ratio of each triangle, defined as `2*inradius / circumradius`.
    q_area_ratio : double NumPy vector
        face area divided by 'ideal area' for each triangle, 
        
        where ideal area is the area of an equilateral triangle whose edge length equals the longest edge in the face.
    area : double NumPy vector
        area of each triangle, in mm^2.
    zeroflag : bool NumPy vector
        flags whether the area a triangle is close to zero.
    edgeflag : int
        flag if any problematic edges. Flag set by bits 'b1b0':
        
        b1=1 if dangling edges found, b0=1 if there exist edges shared by more than two triangles

    """
    # ele is zero-based, int
    if nodes.shape[1] != 3:
        raise TypeError('Mesh must be 3D')
    if ele.shape[1] != 3:
        raise TypeError('Mesh must be a surface mesh')
        
    nnodes = nodes.shape[0]
    nodenumbers = np.arange(nnodes)
    
    # check for nodes not used by any triangles
    tmp = np.isin(nodenumbers, ele)
    if not np.all(tmp):
        print('checkmesh3d_surface: The provided mesh has extra nodes that are not used in the patch element list', flush=1)
        print('Warning: Not all nodes are used in the element connectivity list', flush=1)
        if verbose:
            print(np.nonzero(~tmp)[0])
    
    # check for non-existing nodes used by triangles
    tmp = np.isin(ele, nodenumbers)
    if not np.all(tmp):
        print('checkmesh3d_surface: Some of the triangles are using nodes that are not defined in node list!', flush=1)
        if verbose:
            print(np.nonzero(np.sum(tmp,axis=1)<3)[0])
        raise TypeError('The provided mesh uses node numbers that are not part of the node list!', flush=1)
    
    # check edge integrity
    edgeflag = auxiliary.checkedges(ele)
    
    # check for small triangle areas
    area, zeroflag = auxiliary.check_facearea(nodes, ele)
    
    # check radius ratio for each triangle
    q_radius_ratio = auxiliary.quality_triangle_radius(nodes, ele)
    
    # check area ratio for each triangle
    l1 = np.linalg.norm(nodes[ele[:,0],:] - nodes[ele[:,1],:], axis=1)
    l2 = np.linalg.norm(nodes[ele[:,0],:] - nodes[ele[:,2],:], axis=1)
    l3 = np.linalg.norm(nodes[ele[:,1],:] - nodes[ele[:,2],:], axis=1)
    
    maxl = np.max(np.c_[l1, l2, l3], axis=1)
    ideal_area = np.sqrt(3)*maxl*maxl/4
    q_area_ratio = area/ideal_area
    
    if np.any(zeroflag):
        print('At least one of the patches has a very small area!', flush=1)
        print('Avg Min Max of patch areas: %f %f %f' % (np.mean(area), np.min(area), np.max(area)), flush=1)
    
    return q_radius_ratio, q_area_ratio, area, zeroflag, edgeflag

def CheckMesh3D(elements, nodes, base=1, verbose=False):
    """
    Main function that calculates and checks the quality of a 3D mesh, which can be either a solid or surface mesh
    
    If surface mesh, checkmesh3d_surface() is called
    
    If solid mesh, checkmesh3d_solid() is first used, and checkmesh3d_surface() is subsequently used to check its outer surface

    Parameters
    ----------
    ele : int32 NumPy array
        element list of the mesh, zero-based.
    nodes : double NumPy array
        node locations of the mesh, in mm.
    base : int, optional
        one- or zero-based indexing of the element list. Can be 1, or 0. The default is 1.
    verbose : bool, optional
        whether print the problematic elements to stdout, if any. The default is False.

    Raises
    ------
    TypeError
        if elements and nodes do not define a valid 3D mesh.

    Returns
    -------
    vol : double NumPy vector
        Volume (for solid mesh) or area (for surface mesh) of each element.
    vol_ratio : double NumPy vector
        volume ratio for each tetrahedron in a solid mesh. Returns scalar 0.0 in case of a surface mesh.
    q_area : double NumPy vector
        area ratio for each triangle in a surface mesh. Returns scalar 0.0 in case of a solid mesh.
    status_solid : int
        flags of solid mesh quality, set by bits: 'b3b2b1b0'.
        
        b1 set if small volumes found; b2 set if small volume ratios found; b3 set if faces shared by more than two tetrahedrons found
    status_surface : int
        flags of surface mesh quality, set by bits: 'b3b2b1b0'.
        
        b1 set if edges shared by more than two triangles found; b2 set if dangling edges found; b3 triangles with small area found
        
    See Also
    -------
    :func:`nirfasterff.meshing.checkmesh3d_solid()`, :func:`nirfasterff.meshing.checkmesh3d_surface()`

    """
    if nodes.shape[1] != 3:
        raise TypeError('Mesh must be 3D')
    if elements.min()<0:
        raise TypeError('element list must be non-negative!')
    if base==1:
        ele = np.int32(elements - 1)
        if elements.min()==0:
            print('Warning: min of element list is 0. Maybe it is zero-based?', flush=1)
    elif base==0:
        ele = np.int32(elements)
        if elements.min()==1:
            print('Warning: min of element list is 1. Maybe it is one-based?', flush=1)
    nnpe = ele.shape[1]
    if nnpe==5:
        print('Warning: Ignoring last columnn of elements and treating it as a tetrahedral mesh', flush=1)
        ele = ele[:,:4]
        nnpe = 4
    
    status_surface = 0
    status_solid = 0
    vol = 0.0
    vol_ratio = 0.0
    q_area = 0.0
    q_tri_area_threshold = 0.1
    TetrahedronFailQuality = 0.03
    
    if nnpe == 3:
        print('Checking surface mesh:', flush=1)
        q_radius, q_area, area, zeroflag, edgeflag = checkmesh3d_surface(ele, nodes, verbose)
        vol = area
    elif nnpe == 4:
        print('Checking solid mesh:', flush=1)
        vol, vol_ratio, zeroflag, faceflag = checkmesh3d_solid(ele, nodes, verbose)
    else:
        raise TypeError('CheckMesh3D doesn''t support '+str(nnpe) +' nodes per element!')
    
    if nnpe == 3:
        if edgeflag:
            if edgeflag==1:
                status_surface = status_surface | 2
                print('Some of mesh edges are shared by more than two triangles!', flush=1);
                print('  This can be caused by a non-manifold mesh or a multi-region one.', flush=1);
            if edgeflag==2:
                status_surface = status_surface | 4
                print('Provided surface is not closed:', flush=1);
                print('  At least one of the edges is only shared by only one triangle (it should be two)', flush=1);
            if edgeflag!=1 and edgeflag!=2:
                status_surface = status_surface | 6
                print('Surface mesh is open AND has edges shared by more than 2 triangles!', flush=1)
        
        n_small_tri = np.sum(q_area < q_tri_area_threshold)
        if n_small_tri:
            status_surface = status_surface | 8
            print(' There are %d faces with low quality (q<%f).' % (n_small_tri, q_tri_area_threshold), flush=1)
        
        if (edgeflag or n_small_tri>0) and not verbose:
            print('Set ''verbose=True'' to display the problematic triangles', flush=1)
            
    elif nnpe == 4:
        if faceflag:
            status_solid = status_solid | 8
        if np.any(zeroflag):
            status_solid = status_solid | 2
            print(' There are %d tetrahedrons with small volume.' % np.sum(zeroflag), flush=1)
        n_qfail = np.sum(vol_ratio<TetrahedronFailQuality)
        if n_qfail:
            print(' There are %d tetrahedrons that have a very low quality (q<%f).' % (n_qfail, TetrahedronFailQuality), flush=1)
            status_solid = status_solid | 4
            
        if (faceflag or np.any(zeroflag) or n_qfail>0) and not verbose:
            print('Set ''verbose=True'' to display the problematic tetrahedrons', flush=1)
        
        # get the surface mesh
        ele_surf, nodes_surf = boundfaces(nodes, ele)
        print('----> Checking integrity of the surface of the solid mesh...', flush=1)
        # elements are already zero-based, need to specify here
        _,_,_,_,status_surface = CheckMesh3D(ele_surf, nodes_surf, 0, verbose)
        print('----> Done.', flush=1)
    
    return vol, vol_ratio, q_area, status_solid, status_surface

def CheckMesh2D(elements, nodes, base=1, verbose=False):
    """
    Main function that calculates and checks the quality of a 2D mesh

    Parameters
    ----------
    elements : int32 NumPy array
        element list of the mesh, zero-based.
    nodes : double NumPy array
        node locations of the mesh, in mm.
    base : int, optional
        one- or zero-based indexing of the element list. Can be 1, or 0. The default is 1.
    verbose : bool, optional
        whether print the problematic elements to stdout, if any. The default is False.

    Raises
    ------
    TypeError
        if elements and nodes do not define a valid 2D mesh.

    Returns
    -------
    flag : int
        flags of mesh quality, set by bits: 'b2b1b0'.
        
        b1 set if faulty edges found; b2 set if triangles with small area found
    q_radius_ratio : double NumPy array
        radius ratio of each triangle, defined as `2*inradius / circumradius`.
    area : double NumPy array
        area of each triangle in mesh.

    """
    if not (elements.shape[1]==3 and nodes.shape[1]==3):
        raise TypeError('Mesh must be 2D')
        
    if elements.min()<0:
        raise TypeError('element list must be non-negative!')
    if base==1:
        ele = np.int32(elements - 1)
        if elements.min()==0:
            print('Warning: min of element list is 0. Maybe it is zero-based?', flush=1)
    elif base==0:
        ele = np.int32(elements)
        if elements.min()==1:
            print('Warning: min of element list is 1. Maybe it is one-based?', flush=1)
    
    print('Checking 2D mesh...', flush=1)
    
    # check for small triangle areas
    area, zeroflag = auxiliary.check_facearea(nodes, ele)
    
    # check radius ratio for each triangle
    q_radius_ratio = auxiliary.quality_triangle_radius(nodes, ele)
    
    # check edge integrity
    edgeflag = auxiliary.checkedges(ele)
    if edgeflag:
        print(' Some of the edges of this mesh are shared by more than 2 triangles.', flush=1)
    
    print(' Quality: ranges between 0 and 1, with 1 being best quality:', flush=1)
    print(' Min, Max and Average of triangle quality: %f, %f, %f' % (np.min(q_radius_ratio), np.max(q_radius_ratio), np.mean(q_radius_ratio)), flush=1)
    print(' Min, Max and Average area: %f, %f, %f' % (np.min(area), np.max(area), np.mean(area)), flush=1)
    
    if np.any(zeroflag):
        print(' There are %d triangles whose area is close to zero' % np.sum(zeroflag))
        if verbose:
            print(np.nonzero(zeroflag)[0], flush=1)
    
    ele2 = utils.check_element_orientation_2d(ele+1, nodes)
    if np.any(ele2 - ele):
        print(' Sone elements of this mesh are not oriented CCW!')
    if not edgeflag and not np.all(zeroflag):
        flag = 0
    if edgeflag:
        flag = 2
    if np.any(zeroflag):
        flag = flag | 4
    
    return flag, q_radius_ratio, area
