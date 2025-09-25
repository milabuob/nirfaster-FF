"""
Define the optode class, an instance of which can be either a source or a detector
"""
import numpy as np
from nirfasterff import utils

class optode:
    """
    Class for NIRFASTer optodes, which can be either a group of sources or a group of detectors. 
    
    Note: The field fwhm for sources in the Matlab version has been dropped.
    
    Attributes
    ----------
        fixed: bool like
            whether an optode is fixed. 
            
            If not, it will be moved to one scattering length inside the surface (source) or on the surface (detector).
            
            Default: 0
        num: double NumPy vector
            indexing of the optodes, starting from one (1,2,3,...)
        coord: double NumPy array
            each row is the location of an optode. Unit: mm
        int_func: double NumPy array
            First column is the index (one-based) of the element each optode is in. 
            
            The subsequent columns are the barycentric coordinates (i.e. integration function) in the correponding elements. Size (N, dim+2). 
    
    """
    def __init__(self, coord = []):
        self.fixed = 0
        self.num = []
        self.coord = np.ascontiguousarray(np.atleast_2d(coord), dtype=np.float64)
        self.int_func = []
        
    def move_sources(self, mesh):
        """
        Moves sources to the appropriate locations in the mesh.
        
        For each source, first move it to the closest point on the surface of the mesh, and then move inside by one scattering length along surface normal.
        
        where scattering length is :math:`1/\mu_s'` for stnd and dcs mesh, and :math:`1/\mu_{sx}'` for fluor mesh
        
        Integration functions are also calculated after moving.

        Parameters
        ----------
        mesh : NIRFASTer mesh type
            The mesh on which the sources are installed. Can be a 'stndmesh', 'fluormesh', or 'dcsmesh', either 2D or 3D

        Raises
        ------
        TypeError
            If mesh type is not recognized.
        ValueError
            If mesh.elements does not have 3 or 4 columns, or mesh.dimension is not 2 or 3.

        Returns
        -------
        None.

        """
        
        if mesh.type == 'stnd' or mesh.type == 'dcs':
            mus_eff = mesh.mus
        elif mesh.type == 'fluor':
            mus_eff = mesh.musx
        else:
            raise TypeError('mesh type: '+mesh.type+' unsupported')
        if len(self.coord)==0:
            print('Warning: no optodes to move')
            return
        if len(mus_eff)==0:
            print('Warning: Source cannot be moved. No optical property found.')
            return
        
        scatt_dist = 1.0 / np.mean(mus_eff)
        mesh_size = np.max(mesh.nodes, axis=0) - np.min(mesh.nodes, axis=0)
        if scatt_dist*10. > np.min(mesh_size):
            print('Warning: Mesh is too small for the scattering coefficient given. Minimal mesh size: ' + str(mesh_size.min()) + 'mm. Scattering distance: '
                  + str(scatt_dist) + 'mm. ' + str(mesh_size.min()/10.) + ' mm will be used for scattering distance. \n You might want to ensure that the scale of your mesh and the scattering coefficient are in mm.')
            scatt_dist = mesh_size.min()/10.
        
        if mesh.elements.shape[1] == 4:
            mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
            # Get all faces that touch the boundary
            faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1,2])], 
                          mesh.elements[np.ix_(mask_touch_surface, [0,1,3])],
                          mesh.elements[np.ix_(mask_touch_surface, [0,2,3])],
                          mesh.elements[np.ix_(mask_touch_surface, [1,2,3])]]
            # sort vertex indices to make them comparable
            faces = np.sort(faces)
            # take unique faces
            faces = np.unique(faces, axis=0)
            #take faces where all three vertices are on the boundary
            faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
        elif mesh.elements.shape[1] == 3:
            mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
            # Get all faces that touch the boundary
            faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1])], 
                          mesh.elements[np.ix_(mask_touch_surface, [0,2])],
                          mesh.elements[np.ix_(mask_touch_surface, [1,2])]]
            # sort vertex indices to make them comparable
            faces = np.sort(faces)
            # take unique faces
            faces = np.unique(faces, axis=0)
            # take edges where both two vertices are on the boundary
            faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
        else:
            raise ValueError('mesh.elements has wrong dimensions')
        
        pos1 = np.zeros(self.coord.shape)
        pos2 = np.zeros(self.coord.shape)
        self.int_func = np.zeros((self.coord.shape[0], mesh.dimension+2))
        
        for i in range(self.coord.shape[0]):
            if mesh.dimension == 2:
                # find the closest boundary node
                dist = 1000. * np.ones(mesh.nodes.shape[0])
                dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                r0_ind = np.argmin(dist)
                # find edges including the closest boundary node
                fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                # find closest edge
                dist = np.zeros(fi.shape[0])
                point = np.zeros((fi.shape[0], 2))
                for j in range(fi.shape[0]):
                    dist[j], point[j,:] = utils.pointLineDistance(mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], self.coord[i,:2])
                smallest = np.argmin(dist)
                
                # find norm of that edge
                a = mesh.nodes[fi[smallest, 0], :]
                b = mesh.nodes[fi[smallest, 1], :]
                n = np.array([b[1]-a[1], b[0]-a[0]])
                n = n/np.linalg.norm(n)
                
                # move inside by 1 scattering distance
                pos1[i,:] = point[smallest,:] + n * scatt_dist
                pos2[i,:] = point[smallest,:] - n * scatt_dist
            elif mesh.dimension == 3:
                # find the closest boundary node
                dist = 1000. * np.ones(mesh.nodes.shape[0])
                dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                r0_ind = np.argmin(dist)
                # find edges including the closest boundary node
                fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                # find closest edge
                dist = np.zeros(fi.shape[0])
                point = np.zeros((fi.shape[0], 3))
                for j in range(fi.shape[0]):
                    dist[j], point[j,:] = utils.pointTriangleDistance(np.array([mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], mesh.nodes[fi[j,2],:]]), self.coord[i,:])
                smallest = np.argmin(dist)
                
                # find norm of that edge
                a = mesh.nodes[fi[smallest, 0], :]
                b = mesh.nodes[fi[smallest, 1], :]
                c = mesh.nodes[fi[smallest, 2], :]
                n = np.cross(b-a, c-a)
                n = n/np.linalg.norm(n)
                
                # move inside by 1 scattering distance
                pos1[i,:] = point[smallest,:] + n * scatt_dist
                pos2[i,:] = point[smallest,:] - n * scatt_dist
            else:
                raise ValueError('mesh.dimension should be 2 or 3')
        
        ind, int_func = utils.pointLocation(mesh, pos1)
        in_ind = ind>-1
        self.coord[in_ind,:] = pos1[in_ind,:]
        self.int_func[in_ind,:] = np.c_[ind[in_ind]+1, int_func[in_ind,:]]  # to one-based
        if np.all(in_ind):
            return
        else:
            nan_ind = ~in_ind
            ind2, int_func2 = utils.pointLocation(mesh, pos2[nan_ind,:])
            self.coord[nan_ind,:] = pos2[nan_ind, :]
            self.int_func[nan_ind,:] = np.c_[ind2+1, int_func2] # to one-based
        
        if np.any(ind2==-1):
            print('Warning: Source(s) could not be moved. The mesh structure may be poor.')
        
    
    def move_detectors(self, mesh):
        """
        Moves detector to the appropriate locations in the mesh.
        
        For each detector, first move it to the closest point on the surface of the mesh.
        
        Integration functions are NOT calculated after moving, to be consistent with the Matlab version.

        Parameters
        ----------
        mesh : NIRFASTer mesh type
            The mesh on which the detectors are installed. Can be a 'stndmesh', 'fluormesh', or 'dcsmesh', either 2D or 3D

        Raises
        ------
        ValueError
            If mesh.elements does not have 3 or 4 columns, or mesh.dimension is not 2 or 3.

        Returns
        -------
        None.

        """
        
        if len(self.coord)==0:
            print('Warning: no optodes to move')
            return
        
        if mesh.elements.shape[1] == 4:
            mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
            # Get all faces that touch the boundary
            faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1,2])], 
                          mesh.elements[np.ix_(mask_touch_surface, [0,1,3])],
                          mesh.elements[np.ix_(mask_touch_surface, [0,2,3])],
                          mesh.elements[np.ix_(mask_touch_surface, [1,2,3])]]
            # sort vertex indices to make them comparable
            faces = np.sort(faces)
            # take unique faces
            faces = np.unique(faces, axis=0)
            #take faces where all three vertices are on the boundary
            faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
        elif mesh.elements.shape[1] == 3:
            mask_touch_surface = mesh.bndvtx[np.int32(mesh.elements - 1)].sum(axis=1) > 0
            # Get all faces that touch the boundary
            faces = np.r_[mesh.elements[np.ix_(mask_touch_surface, [0,1])], 
                          mesh.elements[np.ix_(mask_touch_surface, [0,2])],
                          mesh.elements[np.ix_(mask_touch_surface, [1,2])]]
            # sort vertex indices to make them comparable
            faces = np.sort(faces)
            # take unique faces
            faces = np.unique(faces, axis=0)
            # take edges where both two vertices are on the boundary
            faces = faces[mesh.bndvtx[np.int32(faces-1)].sum(axis=1) == mesh.dimension, :] - 1 # convert to zero-based
        else:
            raise ValueError('mesh.elements has wrong dimensions')
        
        for i in range(self.coord.shape[0]):
            if mesh.dimension == 2:
                # find the closest boundary node
                dist = 1000. * np.ones(mesh.nodes.shape[0])
                dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                r0_ind = np.argmin(dist)
                # find edges including the closest boundary node
                fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                # find closest edge
                dist = np.zeros(fi.shape[0])
                point = np.zeros((fi.shape[0], 2))
                for j in range(fi.shape[0]):
                    dist[j], point[j,:] = utils.pointLineDistance(mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], self.coord[i,:2])
                smallest = np.argmin(dist)
                # move detector to the closest point on that edge
                self.coord[i,:] = point[smallest,:]
            elif mesh.dimension == 3:
                # find the closest boundary node
                dist = 1000. * np.ones(mesh.nodes.shape[0])
                dist[mesh.bndvtx>0] = np.linalg.norm(mesh.nodes[mesh.bndvtx>0] - self.coord[i,:], axis=1)
                r0_ind = np.argmin(dist)
                # find edges including the closest boundary node
                fi = np.int32(faces[np.sum(faces==r0_ind, axis=1)>0, :])
                # find closest edge
                dist = np.zeros(fi.shape[0])
                point = np.zeros((fi.shape[0], 3))
                for j in range(fi.shape[0]):
                    dist[j], point[j,:] = utils.pointTriangleDistance(np.array([mesh.nodes[fi[j,0],:], mesh.nodes[fi[j,1],:], mesh.nodes[fi[j,2],:]]), self.coord[i,:])
                smallest = np.argmin(dist)
                # move detector to the closest point on that edge
                self.coord[i,:] = point[smallest,:]
            else:
                raise ValueError('mesh.dimension should be 2 or 3')
        
    
    def touch_sources(self, mesh):
        """
        Recalculate/fill in all other fields based on 'fixed' and 'coord'. 
        
        This is useful when a set of sources are manually added and only the locations are specified.
        
        For non-fixed sources, function 'move_sources' is called, otherwise recalculates integration functions directly
        
        If no source locations are specified, the function does nothing

        Parameters
        ----------
        mesh : NIRFASTer mesh type
            The mesh on which the sources are installed. Can be a 'stndmesh', 'fluormesh', or 'dcsmesh', either 2D or 3D

        Returns
        -------
        None.

        """
        
        if len(self.coord)==0:
            return
        n_src = self.coord.shape[0]
        self.num = np.arange(1, n_src+1, dtype=np.float64)
        if not self.fixed:
            self.move_sources(mesh)
        else:
            ind, int_func = utils.pointLocation(mesh, self.coord)
            self.int_func = np.c_[ind+1, int_func]
    
    def touch_detectors(self, mesh):
        """
        Recalculate/fill in all other fields based on 'fixed' and 'coord'. 
        
        This is useful when a set of detectors are manually added and only the locations are specified.
        
        For non-fixed detectors, function 'move_detectors' is first called, and integration functions are calculated subsequentely.
        
        For fixed detectors, recalculates integration functions directly.
        
        If no detector locations are specified, the function does nothing

        Parameters
        ----------
        mesh : NIRFASTer mesh type
            The mesh on which the sources are installed. Can be a 'stndmesh', 'fluormesh', or 'dcsmesh', either 2D or 3D

        Returns
        -------
        None.

        """
       
        if len(self.coord)==0:
            return
        n_det = self.coord.shape[0]
        self.num = np.arange(1, n_det+1, dtype=np.float64)
        if not self.fixed:
            self.move_detectors(mesh)
        ind, int_func = utils.pointLocation(mesh, self.coord)
        self.int_func = np.c_[ind+1, int_func]
