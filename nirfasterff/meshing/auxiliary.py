"""
Auxiliary functions used for mesh quality check. 

They are unlikely to become useful to an ordinary user, but still documented for completeness

Use with caution: no error checking mechanisms implemented
"""
import numpy as np

def vector_vector_angle(u, v):
    """
    Calculates vector-vector angles, in radian
    
    Each row of u, v is a vector, and the angles are calculated pairwise row by row

    Parameters
    ----------
    u : double NumPy array
        first set of vectors.
    v : double NumPy array
        second set of vectors.

    Returns
    -------
    double NumPy vector
        pairwise vector-vector angles, in radian. Same number of rows as u and v

    """
    u = np.atleast_2d(u)
    v = np.atleast_2d(v)
    cos_theta = np.sum(u*v, axis=1) / (np.linalg.norm(u,axis=1) * np.linalg.norm(v,axis=1))
    # This shouldn't happen but in case of numerical accuracy
    cos_theta[cos_theta<-1.] = -1.
    cos_theta[cos_theta>1.] = 1.
    return np.arccos(cos_theta)

def simpqual(nodes, ele):
    """
    For each tetrahedron, calculates the didehedral angles and returns the smallest sine of them

    Parameters
    ----------
    nodes : double NumPy array
        node locations of the mesh.
    ele : int32 NumPy array
        element list of the mesh, zero-based.

    Returns
    -------
    double NumPy vector
        smallest sine of the dihedral angles for each element. Size (NElements,)
    
    References
    -------
    https://en.wikipedia.org/wiki/Dihedral_angle

    """
    # only the min_sin_didehedral part of the original Matlab version
    # ele is zero-based
    v1 = nodes[ele[:,1],:] - nodes[ele[:,0],:]
    v2 = nodes[ele[:,2],:] - nodes[ele[:,0],:]
    v3 = nodes[ele[:,3],:] - nodes[ele[:,0],:]
    v4 = nodes[ele[:,2],:] - nodes[ele[:,1],:]
    v5 = nodes[ele[:,3],:] - nodes[ele[:,1],:]
    
    n1 = np.cross(v2, v1)
    n2 = np.cross(v1, v3)
    n3 = np.cross(v3, v2)
    n4 = np.cross(v4, v5)
    
    di_angles = np.zeros((ele.shape[0], 6))
    di_angles[:,0] = vector_vector_angle(n1, n2)
    di_angles[:,1] = vector_vector_angle(n1, n3)
    di_angles[:,2] = vector_vector_angle(n2, n3)
    di_angles[:,3] = vector_vector_angle(n1, n4)
    di_angles[:,4] = vector_vector_angle(n2, n4)
    di_angles[:,5] = vector_vector_angle(n3, n4)
    
    di_angles = np.pi - di_angles
    
    return np.min(np.sin(di_angles), axis=1)

def check_tetrahedron_faces(ele):
    """
    Check for faces shared by more than two tetrahedrons

    Parameters
    ----------
    ele : int32 NumPy array
        element list of the mesh, zero-based.

    Returns
    -------
    flag : int
        0 if no faulty faces found, and 2 if faces shared by more than two tetrahedrons are found.

    """
    print('Checking tetrahedral faces..... ', flush=1)
    faces = np.r_[ele[:, [0,1,2]], 
                  ele[:, [0,1,3]],
                  ele[:, [0,2,3]],
                  ele[:, [1,2,3]]]
    
    faces = np.sort(faces)
    # find faces that are not used or used more than twice
    unique_faces, cnt = np.unique(faces, axis=0, return_counts=1)
    bf = (cnt>2) | (cnt==0)
    nbadfaces = np.sum(bf)
    
    if nbadfaces==0:
        # no issues found
        flag = 0
    else:
        # some faces are shared by more than two tetrahedrons: a definite problem
        flag = 2
        print('------------ Invalid solid mesh! ------------', flush=1)
        print('A total %d faces of the mesh are shared by more than two tetrahedrons!'%nbadfaces, flush=1)
        badidx = np.nonzero(bf)[0]
        for i in range(nbadfaces):
            print('Face: %d %d %d' % (unique_faces[badidx[i], 0], unique_faces[badidx[i], 1], unique_faces[badidx[i], 2]), flush=1)
            print('Tets:', flush=1)
            junk = np.isin(ele, unique_faces[badidx[i],:])
            badtets = np.nonzero(np.sum(junk, axis=1)==3)[0]
            print(badtets, flush=1)
    print('Done', flush=1)
    return flag

def checkedges(ele):
    """
    Check for orphan edges and edges shared by more than two triangles

    Parameters
    ----------
    ele : int32 NumPy array
        element list of the mesh, zero-based.

    Returns
    -------
    flag : int
        0 if no errors found; 1 if edges shared by more than two triangles found; 2 if dangling edges found; 3 if both errors found.

    """
    flag = 0
    edges = np.r_[ele[:, [0,1]], 
                  ele[:, [0,2]],
                  ele[:, [1,2]]]
    edges = np.sort(edges)
    # check for orphan edges and edges shared by more than two triangles
    unique_edges, cnt = np.unique(edges, axis=0, return_counts=1)
    
    orphan = np.nonzero(cnt==0)[0]
    reused = np.nonzero(cnt>2)[0]
    
    if len(orphan)>0:
        flag = 2
        print('Orphan edges found:', flush=1)
        print(orphan)
    if len(reused)>0:
        flag  += 1
        print('Edges shared by more than two triangles found:')
        print(reused)
    return flag

def check_facearea(nodes, ele):
    """
    Calculates the areas of each face, and check if they are close to zero
    
    Close to zero defined as 1e6 of the max span of the mesh

    Parameters
    ----------
    nodes : double NumPy array
        node locations of the mesh.
    ele : int32 NumPy array
        element list of the mesh, zero-based.

    Returns
    -------
    area : double NumPy vector
        areas of each face. Size (NElements,)
    zeroflag : bool NumPy vector
        flags of whether the area is close to zero, for each face. Size (NElements,)

    """
    # ele is zero-based
    span = np.max(nodes, axis=0) - np.min(nodes, axis=0)
    tiny = span.max()*1e-6
    u = nodes[ele[:,2],:] - nodes[ele[:,0],:]
    v = nodes[ele[:,1],:] - nodes[ele[:,0],:]
    area = 0.5*np.linalg.norm(np.cross(u,v), axis=1)
    zeroflag = area<=tiny
    return area, zeroflag

def quality_triangle_radius(nodes, ele):
    """
    Radius ratio: 2*inradius / circumradius
    
    Value between 0 and 1. Equals 1 only when a triangle is equilateral

    Parameters
    ----------
    nodes : double NumPy array
        node locations of the mesh.
    ele : int32 NumPy array
        element list of the mesh, zero-based.

    Returns
    -------
    double NumPy vector
        radius ratios for each triangle. Size (NElements,)
    
    References
    -------
    https://en.wikibooks.org/wiki/Trigonometry/Circles_and_Triangles/The_Incircle

    """
    # calculates quality of triangular meshes using radius ratio method.
    # ele is zero-based
    a = np.linalg.norm(nodes[ele[:,1],:] - nodes[ele[:,0],:], axis=1)
    b = np.linalg.norm(nodes[ele[:,2],:] - nodes[ele[:,0],:], axis=1)
    c = np.linalg.norm(nodes[ele[:,2],:] - nodes[ele[:,1],:], axis=1)
    # inradius and circumradius: https://en.wikibooks.org/wiki/Trigonometry/Circles_and_Triangles/The_Incircle
    r = 0.5 * np.sqrt((b+c-a) * (c+a-b) * (a+b-c) / (a+b+c))
    R = a*b*c / np.sqrt((a+b+c)*(b+c-a)*(c+a-b)*(a+b-c))
    return 2.0*r / R


