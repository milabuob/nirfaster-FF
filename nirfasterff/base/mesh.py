"""
Define the parent mesh class
"""
import numpy as np
# from data import FDdata
from .data import meshvol
from .optodes import optode
import copy
import os
from nirfasterff import utils
from nirfasterff import meshing
from nirfasterff import forward
from nirfasterff import inverse
from nirfasterff import visualize
import scipy.io as sio

class mesh():
    '''
    Parent class of all meshes. Here defines the attributes and functions shared by all mesh types.
    
    Attributes:
    -----------
    name: str
        name of the mesh. Default: 'EmptyMesh'
    nodes: double NumPy array
        locations of nodes in the mesh. Unit: mm. Size (NNodes, dim)
    bndvtx: double NumPy array
        indicator of whether a node is at boundary (1) or internal (0). Size (NNodes,)
    ri: double NumPy array 
        refractive index at each node. Size (NNodes,)
    elements: double NumPy array 
        triangulation (tetrahedrons or triangles) of the mesh, Size (NElements, dim+1)
        
        Row i contains the indices of the nodes that form tetrahedron/triangle i
        
        One-based indexing for direct interoperatability with the Matlab version
    region: double NumPy array 
        region labeling of each node. Starting from 1. Size (NNodes,)
    source: nirfasterff.base.optode 
        information about the sources
    meas: nirfasterff.base.optode
        information about the the detectors
    link: int32 NumPy array 
        list of source-detector pairs, i.e. channels. Size (NChannels,3)
        
        First column: source; Second column: detector; Third column: active (1) or not (0)
    c: double NumPy array 
        light speed (mm/sec) at each node.  Size (NNodes,). Defined as c0/ri, where c0 is the light speed in vacuum
    ksi: double NumPy array 
        photon fluence rate scale factor on the mesh-outside_mesh boundary as derived from Fresenel's law. Size (NNodes,)
    element_area: double NumPy array 
        volume/area (mm^3 or mm^2) of each element. Size (NElements,)
    support: double NumPy array 
        total volume/area of all the elements each node belongs to. Size (NNodes,)
    vol: nirfasterff.base.meshvol 
        object holding information for converting between mesh and volumetric space.
    '''
    def __init__(self):
        self.name = 'EmptyMesh'
        self.nodes = []
        self.bndvtx = []
        self.ri = []
        self.elements = []
        self.dimension = []
        self.region = []
        self.source = optode()
        self.meas = optode()
        self.link = []
        self.c = []
        self.ksi = []
        self.element_area = []
        self.support = []
        self.vol = meshvol()
    
    def touch_optodes(self):
        """
        Moves all optodes (if non fixed) and recalculate the integration functions (i.e. barycentric coordinates). 
        
        See :func:`~nirfasterff.base.optodes.optode.touch_sources()` and :func:`~nirfasterff.base.optodes.optode.touch_detectors()` for details

        Returns
        -------
        None.

        """
        # make sure the optodes sit correctly: moved if needed, calculate the int func
        print('touching sources', flush=1)
        self.source.touch_sources(self)
        print('touching detectors', flush=1)
        self.meas.touch_detectors(self)
    
    def isvol(self):
        """
        Check if convertion matrices between mesh and volumetric spaces are calculated

        Returns
        -------
        bool
            True if attribute `.vol` is calculated, False if not.

        """
        
        if len(self.vol.xgrid):
            return True
        else:
            return False
    
    def gen_intmat(self, xgrid, ygrid, zgrid=[]):
        """
        Calculate the information needed to convert data between mesh and volumetric space, specified by x, y, z (if 3D) grids.
        
        All grids must be uniform. The results will from a nirfasterff.base.meshvol object stored in field .vol
        
        If field .vol already exists, it will be calculated again, and a warning will be thrown

        Parameters
        ----------
        xgrid : double NumPy array
            x grid in mm.
        ygrid : double NumPy array
            x grid in mm.
        zgrid : double NumPy array, optional
            x grid in mm. Leave empty for 2D meshes. The default is [].
        
        Raises
        ------
        ValueError
            if grids not uniform, or zgrid empty for 3D mesh

        Returns
        -------
        None.

        """

        xgrid = np.float64(np.array(xgrid).squeeze())
        ygrid = np.float64(np.array(ygrid).squeeze())
        zgrid = np.float64(np.array(zgrid).squeeze())
        tmp = np.diff(xgrid)
        if np.any(tmp-tmp[0]):
            raise ValueError('xgrid must be uniform')
        tmp = np.diff(ygrid)
        if np.any(tmp-tmp[0]):
            raise ValueError('ygrid must be uniform')
        if self.dimension ==3 and np.size(zgrid)==0:
            raise ValueError('zgrid must be non-empty for 3D mesh')
        if len(zgrid)>0:
            tmp = np.diff(zgrid)
            if np.any(tmp-tmp[0]):
                raise ValueError('zgrid must be uniform')
            
        if self.isvol():
            print('Warning: recalculating intmat', flush=1)

        self.vol.xgrid = xgrid
        self.vol.ygrid = ygrid
        if len(zgrid)>0:
            self.vol.zgrid = zgrid
            self.vol.res = np.array([xgrid[1]-xgrid[0], ygrid[1]-ygrid[0], zgrid[1]-zgrid[0]])
        else:
            self.vol.res = np.array([xgrid[1]-xgrid[0], ygrid[1]-ygrid[0]])
        
        
        self.vol.nn, self.vol.gridinmesh, self.vol.meshingrid, self.vol.mesh2grid, self.vol.grid2mesh = utils.gen_intmat_impl(self, xgrid, ygrid, zgrid)
        
    def voxelize(self, resolution):
        '''
        Calculate the information needed to convert data between mesh and volumetric space by specifying only the resolution.
        Volumetric space defined by the boundaries of the mesh. A short hand of .gen_intmat if wish to voxelize the entire mesh.

        Parameters
        ----------
        resolution : float or float array-like
            resolution of the voxel space in mm.
            if scalar, same resolution in all directions, otherwise it is defined for each direction

        Returns
        -------
        None.

        '''
        dim = self.dimension
        lowerlimit = self.nodes.min(axis=0)
        upperlimit = self.nodes.max(axis=0)
        zgrid = []
        if np.size(resolution)==1: # same resolution in all directions
            xgrid = np.arange(np.floor(lowerlimit[0]), np.ceil(upperlimit[0])+1, resolution)
            ygrid = np.arange(np.floor(lowerlimit[1]), np.ceil(upperlimit[1])+1, resolution)
            if dim==3:
                zgrid = np.arange(np.floor(lowerlimit[2]), np.ceil(upperlimit[2])+1, resolution)
        else:
            xgrid = np.arange(np.floor(lowerlimit[0]), np.ceil(upperlimit[0])+1, resolution[0])
            ygrid = np.arange(np.floor(lowerlimit[1]), np.ceil(upperlimit[1])+1, resolution[1])
            if dim==3:
                zgrid = np.arange(np.floor(lowerlimit[2]), np.ceil(upperlimit[2])+1, resolution[2])
        self.gen_intmat(xgrid, ygrid, zgrid)
        
    def nntogrid(self, field):
        '''
        Convert attributed specified by 'field' to voxel space using nearest-neighbor interpolation
        For example:
            region = mesh.nntogrid('region')
        Note that mesh.gen_intmat() OR mesh.voxelize() must be called prior to this function

        Parameters
        ----------
        field : str
            an attribute of the mesh that is to be interpolated to voxel space.

        Raises
        ------
        TypeError
            if 'field' is not a string.

        Returns
        -------
        result : NumPy array
            mesh.field interpolated to voxel space. Size defined by mesh.vol
            See also: :func:`~nirfasterff.utils.nntogrid`

        '''
        if not type(field) == str:
            raise TypeError('field must be a string')
        result = eval('utils.nntogrid(self, self.' + field + ')')
        return result
    
    def plot(self, data=None, selector=None, alpha=1, cmap='hot', clim=None):
        '''
        Visualize the mesh. If no data is given, shows the mesh itself with regions represented in different colors.
        Alternatively, if mesh space data (e.g. fluence) is given, function visualizes data on the mesh
        
        Intersections are specified using 'selector' argument. If not given, the outer shell is shown.

        See :func:`~nirfasterff.visualize.plotimage()` and :func:`~nirfasterff.visualize.plot3dmesh_v2()` for more details.

        Parameters
        ----------
        data : NumPy array, optional
            Data to be represented on the mesh, with size (NNode,). If not given, mesh.region is used.
        selector : str, optional
            Specifies the intersection at which the data will be plotted, e.g. 'x>50', or '(x>50) | (y<100)', or 'x + y + z < 200'.
            
            Note that "=" is not supported. When "|" or "&" are used, make sure that all conditions are put in parantheses separately
            
            If not specified, function plots the outermost shell of the mesh.
        alpha : float, optional
            opacity, between 0-1. The default is 1.
        cmap : str, optional
            Colormap used to visualize the data. The default is 'hot'.
        clim : Array-like, optional
            colorlimit of the plot in format [cmin, cmax]. The default is None.

        Returns
        -------
        None.

        '''
        if self.dimension==2:
            if np.all(data==None):
                visualize.plotimage(self, self.region, cmap='viridis')
            else:
                visualize.plotimage(self, data, cmap)
        else:
            if np.all(data==None):
                visualize.plot3dmesh_v2(self, self.region, selector, alpha, cmap='viridis')
            else:
                visualize.plot3dmesh_v2(self, data, selector, alpha, cmap, clim)
    
    def plotvol(self, data, bnd=False, cmap='hot', clim=None, surfcnt=25, surfalpha=0.1):
        '''
        Renders volumetric data which is represented in the voxel space defined in mesh.vol.
        Can be used to render e.g. voxels-space fluence or Jacobian bananas
                
        When 'bnd' is set to True, the outmost surface of the mesh is also plotted as a wire frame.
        This can take a few seconds when plotting large meshes.
        
        See :func:`~nirfasterff.visualize.plotvol()` for details.


        Parameters
        ----------
        data : NumPy array
            Can either be a 3D volume consistent with the volumetric space, or its vetorized representation (F order).
        bnd : bool, optional
            If true, also plots the outer boundadry of the mesh as a wire frame. The default is False.
        cmap : str, optional
            colormap to use. The default is 'hot'.
        clim : array-like, optional
            colorlimit of the plot in format [cmin, cmax]. The default is None.
        surfcnt : int, optional
            number of isosurfaces used in volume rendering. The default is 25.
        surfalpha : float, optional
            opacity of the isosurfaces in volume rendering (0-1). The default is 0.1.

        Raises
        ------
        TypeError
            If `mesh.vol` is empty.

        Returns
        -------
        None.

        '''
        if not self.isvol():
            raise TypeError('Volumetric space not defined. Run mesh.gen_intmat() or mesh.voxelize() first.')
        visualize.plotvol(self, data, bnd, cmap, clim, surfcnt, surfalpha)
        
