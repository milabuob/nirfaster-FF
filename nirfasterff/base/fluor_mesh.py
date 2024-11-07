"""
Define the fluorescence mesh class
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
import scipy.io as sio

class fluormesh:
    """
    Main class for fluorescence mesh. The methods should cover most of the commonly-used functionalities
    
    Attributes
    ----------
        name: str
            name of the mesh. Default: 'EmptyMesh'
        nodes: double NumPy array
            locations of nodes in the mesh. Unit: mm. Size (NNodes, dim)
        bndvtx: double NumPy array
            indicator of whether a node is at boundary (1) or internal (0). Size (NNodes,)
        type: str
            type of the mesh. It is always 'fluor'.
        muax: double NumPy array
            intrisic absorption coefficient (mm^-1) at excitation wavelength at each node. Size (NNodes,)
        muam: double NumPy array
            intrisic absorption coefficient (mm^-1) at emission wavelength at each node. Size (NNodes,)
        muaf: double NumPy array
            intrisic absorption coefficient (mm^-1) of the fluorophores at each node. Size (NNodes,)
        kappax: double NumPy array
            intrisic diffusion coefficient (mm) at excitation wavelength at each node. Size (NNodes,). Defined as 1/(3*(muax + musx))
        kappam: double NumPy array
            intrisic diffusion coefficient (mm) at emission wavelength at each node. Size (NNodes,). Defined as 1/(3*(muam + musm))
        ri: double NumPy array 
            refractive index at each node. Size (NNodes,)
        musx: double NumPy array
            intrisic reduced scattering coefficient (mm^-1) at excitation wavelength at each node. Size (NNodes,)
        musm: double NumPy array
            intrisic reduced scattering coefficient (mm^-1) at emission wavelength at each node. Size (NNodes,)
        eta: double NumPy array
            quantum efficiency (a.u.) of the fluorophores. Size (NNodes,)
        tau: double NumPy array
            time coefficient (second) of the fluorophores. Size (NNodes,)
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
        
    """
   
    def __init__(self):
        self.name = 'EmptyMesh'
        self.nodes = []
        self.bndvtx = []
        self.type = 'fluor'
        self.muax = []
        self.kappax = []
        self.musx = []
        self.ri = []
        self.muam = []
        self.kappam = []
        self.musm = []
        self.muaf = []
        self.eta = []
        self.tau = []
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
    
    def from_copy(self, mesh):
        """
        Deep copy all fields from another mesh.

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            the mesh to copy from.

        Returns
        -------
        None.
        """
        
        self.name = copy.deepcopy(mesh.name)
        self.nodes = copy.deepcopy(mesh.nodes)
        self.bndvtx = copy.deepcopy(mesh.bndvtx)
        self.type = 'fluor'
        self.muax = copy.deepcopy(mesh.muax)
        self.kappax = copy.deepcopy(mesh.kappax)
        self.musx = copy.deepcopy(mesh.musx)
        self.ri = copy.deepcopy(mesh.ri)
        self.muam = copy.deepcopy(mesh.muam)
        self.kappam = copy.deepcopy(mesh.kappam)
        self.musm = copy.deepcopy(mesh.musm)
        self.muaf = copy.deepcopy(mesh.muaf)
        self.eta = copy.deepcopy(mesh.eta)
        self.tau = copy.deepcopy(mesh.tau)
        self.elements = copy.deepcopy(mesh.elements)
        self.dimension = copy.deepcopy(mesh.dimension)
        self.region = copy.deepcopy(mesh.region)
        self.source = copy.deepcopy(mesh.source)
        self.meas = copy.deepcopy(mesh.meas)
        self.link = copy.deepcopy(mesh.link)
        self.c = copy.deepcopy(mesh.c)
        self.ksi = copy.deepcopy(mesh.ksi)
        self.element_area = copy.deepcopy(mesh.element_area)
        self.support = copy.deepcopy(mesh.support)
        self.vol = copy.deepcopy(mesh.vol)
    
    def from_file(self, file):
        """
        Read from classic NIRFAST mesh (ASCII) format, not checking the correctness of the loaded integration functions.
        
        All fields after loading should be directly compatible with Matlab version.

        Parameters
        ----------
        file : str
            name of the mesh. Any extension will be ignored.

        Returns
        -------
        None.
        
        Examples
        -------
        >>> mesh = nirfasterff.base.fluormesh()
        >>> mesh.from_file('meshname')

        """
        
        if '.mat' in file:
            raise TypeError('Error: seemingly you are trying to load a mat file. Please use .from_mat method instead')
        # clear data
        self.__init__()
        file = os.path.splitext(file)[0] # in case an extension is accidentally included
        self.name = os.path.split(file)[1]
        # Read the nodes
        if os.path.isfile(file + '.node'):
            fullname = file + '.node'
            tmp = np.genfromtxt(fullname, delimiter='\t', dtype=np.float64)
            tmp = np.atleast_2d(tmp)
            self.bndvtx = np.ascontiguousarray(tmp[:,0])
            self.nodes = np.ascontiguousarray(tmp[:,1:])
            self.dimension = tmp.shape[1]-1
        else:
            raise ValueError('Error: ' + file + '.node file is not present')

        # Read the parameters
        if os.path.isfile(file + '.param'):
            fullname = file + '.param'
            with open(fullname, 'r') as paramfile:
                header = paramfile.readline()
            if ord(header[0])>=48 and ord(header[0])<=57:
                raise ValueError('Error: header missing in .param file. You are probably using the old format, which is no longer supported')
            elif 'fluor' not in header:
                raise ValueError('Error: the mesh you are trying to load is not a fluorescence mesh')
            else:
                tmp = np.genfromtxt(fullname, skip_header=1, dtype=np.float64)
                tmp = np.atleast_2d(tmp)
                self.muax = np.ascontiguousarray(tmp[:,0])
                self.kappax = np.ascontiguousarray(tmp[:,1])
                self.ri = np.ascontiguousarray(tmp[:,2])
                self.musx = (1./self.kappax)/3. - self.muax
                self.muam = np.ascontiguousarray(tmp[:,3])
                self.kappam = np.ascontiguousarray(tmp[:,4])
                self.musm = (1./self.kappam)/3. - self.muam
                self.muaf = np.ascontiguousarray(tmp[:,5])
                self.eta = np.ascontiguousarray(tmp[:,6])
                self.tau = np.ascontiguousarray(tmp[:,7])
        else:
            raise ValueError('Error: ' + file + '.param file is not present')
        # Read the elements
        if os.path.isfile(file + '.elem'):
            fullname = file + '.elem'
            ele = np.atleast_2d(np.genfromtxt(fullname, delimiter='\t', dtype=np.float64))
            ele = np.sort(ele)
            if self.dimension==2:
                self.elements = np.ascontiguousarray(utils.check_element_orientation_2d(ele, self.nodes))
            else:
                self.elements = np.ascontiguousarray(ele)
            if ele.shape[1]-1 != self.dimension:
                print('Warning: nodes and elements seem to have incompatable dimentions. Are you using an old 2D mesh?')
        else:
            raise ValueError('Error: ' + file + '.elem file is not present')

        # Read the region information
        if os.path.isfile(file + '.region'):
            fullname = file + '.region'
            self.region = np.ascontiguousarray(np.genfromtxt(fullname, dtype=np.float64))                    
        else:
            raise ValueError('Error: ' + file + '.region file is not present')

        # Read the source file
        if not os.path.isfile(file + '.source'):
            print('Warning: source file is not present')
        else:
            fullname = file + '.source'
            errorflag = False
            with open(fullname, 'r') as srcfile:
                hdr1 = srcfile.readline()
                hdr2 = srcfile.readline()
            if ord(hdr1[0])>=48 and ord(hdr1[0])<=57:
                print('WARNING: header missing in .source file. You are probably using the old format, which is no longer supported.\nSource not loaded')
                errorflag = True
            elif 'num' not in hdr1 and 'num' not in hdr2:
                print('WARNING: Incorrect or old header format.\nSource not loaded')
                errorflag = True
            elif 'fixed' in hdr1:
                fixed = 1
                N_hdr = 2
                hdr = hdr2.split()
            else:
                fixed = 0
                N_hdr = 1
                hdr = hdr1.split()
            if not errorflag:
                tmp = np.genfromtxt(fullname, skip_header=N_hdr, dtype=np.float64)
                tmp = np.atleast_2d(tmp)
                src = optode()
                src.fixed = fixed
                src.num = tmp[:, hdr.index('num')]
                if 'z' not in hdr:
                    src.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')]]
                else:
                    src.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')], tmp[:, hdr.index('z')]]
                    if self.dimension==2:
                        print('Warning: Sources are 3D, mesh is 2D.')
                if 'fwhm' in hdr:
                    fwhm = tmp[:, hdr.index('fwhm')]
                    if np.any(fwhm):
                        print('Warning: Only point sources supported. Ignoring field fwhm')
                if 'ele' in hdr:
                    if 'ip1' in hdr and 'ip2' in hdr and 'ip3' in hdr:
                        src.int_func = np.c_[tmp[:, hdr.index('ele')], tmp[:, hdr.index('ip1')], tmp[:, hdr.index('ip2')], tmp[:, hdr.index('ip3')]]
                        if 'ip4' in hdr:
                            src.int_func = np.c_[src.int_func, tmp[:, hdr.index('ip4')]]
                            if self.dimension==2:
                                print('Warning: Sources ''int_func'' are 3D, mesh is 2D.')
                        else:
                            if self.dimension==3:
                                print('Warning: Sources ''int_func'' are 2D, mesh is 3D. Will recalculate')
                                src.int_func = []
                    else:
                        print('Warning: source int_func stored in wrong format or missing. Will recalculate')
                
                if src.fixed==1  or len(src.int_func)>0:
                    if src.fixed==1:
                        print('Fixed sources')
                    if len(src.int_func)>0:
                        print('Sources integration functions loaded')
                else:
                    # non-fixed sources and no int_func loaded. Let's move the sources by one scattering length now
                    print('Moving Sources', flush = 1)
                    src.move_sources(self)
                if len(src.int_func)==0:
                    print('Calculating sources integration functions', flush = 1)
                    ind, int_func = utils.pointLocation(self, src.coord)
                    src.int_func = np.c_[ind+1, int_func]
                src.int_func[src.int_func[:,0]==0, 0] = np.nan
                self.source = src
                if not np.all(np.isfinite(src.int_func[:,0])):
                    print('Warning: some sources might be outside the mesh')
        # Read the detector file
        if not os.path.isfile(file + '.meas'):
            print('Warning: detector file is not present')
        else:
            fullname = file + '.meas'
            errorflag = False
            with open(fullname, 'r') as srcfile:
                hdr1 = srcfile.readline()
                hdr2 = srcfile.readline()
            if ord(hdr1[0])>=48 and ord(hdr1[0])<=57:
                print('WARNING: header missing in .meas file. You are probably using the old format, which is no longer supported.\nDetector not loaded')
                errorflag = True
            elif 'num' not in hdr1 and 'num' not in hdr2:
                print('WARNING: Incorrect or old header format.\nDetector not loaded')
                errorflag = True
            elif 'fixed' in hdr1:
                fixed = 1
                N_hdr = 2
                hdr = hdr2.split()
            else:
                fixed = 0
                N_hdr = 1
                hdr = hdr1.split()
            if not errorflag:
                tmp = np.genfromtxt(fullname, skip_header=N_hdr, dtype=np.float64)
                tmp = np.atleast_2d(tmp)
                det = optode()
                det.fixed = fixed
                det.num = tmp[:, hdr.index('num')]
                if 'z' not in hdr:
                    det.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')]]
                else:
                    det.coord = np.c_[tmp[:, hdr.index('x')], tmp[:, hdr.index('y')], tmp[:, hdr.index('z')]]
                    if self.dimension==2:
                        print('Warning: Detectors are 3D, mesh is 2D.')
                if 'ele' in hdr:
                    if 'ip1' in hdr and 'ip2' in hdr and 'ip3' in hdr:
                        det.int_func = np.c_[tmp[:, hdr.index('ele')], tmp[:, hdr.index('ip1')], tmp[:, hdr.index('ip2')], tmp[:, hdr.index('ip3')]]
                        if 'ip4' in hdr:
                            det.int_func = np.c_[det.int_func, tmp[:, hdr.index('ip4')]]
                            if self.dimension==2:
                                print('Warning: Detectors ''int_func'' are 3D, mesh is 2D.')
                        else:
                            if self.dimension==3:
                                print('Warning: Detectors ''int_func'' are 2D, mesh is 3D. Will recalculate')
                                det.int_func = []
                    else:
                        print('Warning: detector int_func stored in wrong format. Will recalculate')
                
                if det.fixed==1  or len(det.int_func)>0:
                    if det.fixed==1:
                        print('Fixed detectors')
                    if len(det.int_func)>0:
                        print('Detectors integration functions loaded')
                else:
                    # non-fixed sources and no int_func loaded. Let's move the sources by one scattering length now
                    print('Moving Detectors', flush = 1)
                    det.move_detectors(self)
                if len(det.int_func)==0:
                    print('Calculating detectors integration functions', flush = 1)
                    ind, int_func = utils.pointLocation(self, det.coord)
                    det.int_func = np.c_[ind+1, int_func]
                det.int_func[det.int_func[:,0]==0, 0] = np.nan
                self.meas = det
                if not np.all(np.isfinite(det.int_func[:,0])):
                    print('Warning: some detectors might be outside the mesh')
        # load link list
        if os.path.isfile(file + '.link'):
            fullname = file + '.link'
            with open(fullname, 'r') as linkfile:
                header = linkfile.readline()
            if ord(header[0])>=48 and ord(header[0])<=57:
                print('Warning: header missing in .link file. You are probably using the old format, which is no longer supported')
            else:
                self.link = np.atleast_2d(np.genfromtxt(fullname, skip_header=1, dtype=np.int32))
        else:
            print('Warning: link file is not present')
        # Speed of light in medium
        c0 = 299792458000.0 # mm/s
        self.c = c0/self.ri
        # Set boundary coefficient using definition of baundary attenuation A using the Fresenel's law; Robin type
        n_air = 1.
        n = self.ri/n_air
        R0 = ((n-1.)**2)/((n+1.)**2)
        theta = np.arcsin(1.0/n)
        A = (2.0/(1.0 - R0) -1. + np.abs(np.cos(theta))**3) / (1.0 - np.abs(np.cos(theta))**2)
        self.ksi = 1.0 / (2*A)
        # area and support for each element
        self.element_area = utils.cpulib.ele_area(self.nodes, self.elements)
        self.support = utils.cpulib.mesh_support(self.nodes, self.elements, self.element_area)
    
    def from_mat(self, matfile, varname = None):
        """
        Read from Matlab .mat file that contains a NIRFASTer mesh struct. All fields copied as is without error checking.

        Parameters
        ----------
        matfile : str
            name of the .mat file to load. Use of extension is optional.
        varname : str, optional
            if your .mat file contains multiple variables, use this argument to specify which one to load. The default is None.
            
            When `varname==None`, `matfile` should contain exatly one structure, which is a NIRFASTer mesh, or the function will do nothing

        Returns
        -------
        None.

        """
        if type(matfile) != str:
            raise TypeError('argument 1 must be a string!')
        if varname != None and type(varname) != str:
            raise TypeError('argument 2 must be a string!')
        
        try:
            tmp = sio.loadmat(matfile, struct_as_record=False, squeeze_me=True)
        except:
            raise TypeError('Failed to load Matlab file ' + matfile + '!')
        
        if varname != None:
            try:
                mesh = tmp[varname]
            except:
                raise TypeError('Cannot load mesh ' + varname + ' from mat file ' + matfile)
        else:
            allkeys = list(tmp.keys())
            is_struct = [type(tmp[key])==sio.matlab._mio5_params.mat_struct for key in allkeys]
            if sum(is_struct) != 1:
                raise TypeError('There must be precisely one struct in the mat file, if "varname" is not provided')
            else:
                varname = allkeys[is_struct.index(True)]
                mesh = tmp[varname]
                
        if mesh.type != 'fluor':
            raise TypeError('mesh type must be fluorescence')
        # Now let's copy the data
        self.__init__()
        self.name = mesh.name
        self.nodes = np.atleast_2d(np.ascontiguousarray(mesh.nodes, dtype=np.float64))
        self.bndvtx = np.ascontiguousarray(mesh.bndvtx, dtype=np.float64)
        self.muax = np.ascontiguousarray(mesh.muax, dtype=np.float64)
        self.kappax = np.ascontiguousarray(mesh.kappax, dtype=np.float64)
        self.musx = np.ascontiguousarray(mesh.musx, dtype=np.float64)
        self.ri = np.ascontiguousarray(mesh.ri, dtype=np.float64)
        self.muam = np.ascontiguousarray(mesh.muam, dtype=np.float64)
        self.kappam = np.ascontiguousarray(mesh.kappam, dtype=np.float64)
        self.musm = np.ascontiguousarray(mesh.musm, dtype=np.float64)
        self.elements = np.atleast_2d(np.ascontiguousarray(mesh.elements, dtype=np.float64))
        self.dimension = mesh.dimension
        self.region = np.ascontiguousarray(mesh.region, dtype=np.float64)
        self.ksi = np.ascontiguousarray(mesh.ksi, dtype=np.float64)
        self.c = np.ascontiguousarray(mesh.c, dtype=np.float64)
        self.element_area = np.ascontiguousarray(mesh.element_area, dtype=np.float64)
        self.support = np.ascontiguousarray(mesh.support, dtype=np.float64)
        allfields = mesh._fieldnames
        if 'source' in allfields:
            self.source.fixed = mesh.source.fixed
            self.source.num = np.ascontiguousarray(mesh.source.num, dtype=np.int32)
            self.source.coord = np.ascontiguousarray(mesh.source.coord, dtype=np.float64)
            self.source.int_func = np.ascontiguousarray(mesh.source.int_func, dtype=np.float64)
        else:
            print('Warning: sources are not present in mesh')
        if 'meas' in allfields:
            self.meas.fixed = mesh.meas.fixed
            self.meas.num = np.ascontiguousarray(mesh.meas.num, dtype=np.int32)
            self.meas.coord = np.ascontiguousarray(mesh.meas.coord, dtype=np.float64)
            self.meas.int_func = np.ascontiguousarray(mesh.meas.int_func, dtype=np.float64)
        else:
            print('Warning: detectors are not present in mesh')
        if 'link' in allfields:
            self.link = np.ascontiguousarray(mesh.link, dtype=np.int32)
        else:
            print('Warning: link is not present in mesh')
        if 'vol' in allfields:
            self.vol.xgrid = mesh.vol.xgrid
            self.vol.ygrid = mesh.vol.ygrid
            self.vol.zgrid = mesh.vol.zgrid
            self.vol.mesh2grid = mesh.vol.mesh2grid
            self.vol.gridinmesh = mesh.vol.gridinmesh
            self.vol.res = mesh.vol.res
            self.vol.grid2mesh = mesh.vol.grid2mesh
            self.vol.meshingrid = mesh.vol.meshingrid
            
    def from_solid(self, ele, nodes, prop = None, src = None, det = None, link = None):
        """
        Construct a NIRFASTer mesh from a 3D solid mesh generated by a mesher. Similar to the solidmesh2nirfast function in Matlab version.
        
        Can also set the optical properties and optodes if supplied

        Parameters
        ----------
        ele : int/double NumPy array
            element list in one-based indexing. If four columns, all nodes will be labeled as region 1
            
            If five columns, the last column will be used for region labeling.
        nodes : double NumPy array
            node locations in the mesh. Unit: mm. Size (NNodes,3).
        prop : double NumPy array, optional
            If not `None`, calls `fluormesh.set_prop()` and sets the optical properties in the mesh. The default is None.
            
            See :func:`~nirfasterff.base.fluor_mesh.fluormesh.set_prop()` for details. 
        src : nirfasterff.base.optode, optional
            If not `None`, sets the sources and moves them to the appropriate locations. The default is None.
            
            See :func:`~nirfasterff.base.optodes.optode.touch_sources()` for details.
        det : nirfasterff.base.optode, optional
            If not `None`, sets the detectors and moves them to the appropriate locations. The default is None.
            
            See :func:`~nirfasterff.base.optodes.optode.touch_detectors()` for details.
        link : int32 NumPy array, optional
            If not `None`, sets the channel information. Uses one-based indexing. The default is None.
            
            Each row represents a channel, in the form of `[src, det, active]`, where `active` is 0 or 1
            
            If `link` contains only two columns, all channels are considered active.

        Returns
        -------
        None.

        """
        self.__init__()
        num_nodes = nodes.shape[0]
        self.nodes = np.ascontiguousarray(nodes, dtype=np.float64)
        if ele.shape[1] == 4:
            # no region label
            self.elements = np.ascontiguousarray(np.sort(ele,axis=1), dtype=np.float64)
            self.region = np.ones(num_nodes)
        elif ele.shape[1] == 5:
            self.region = np.zeros(num_nodes)
            # convert element label to node label
            labels = np.unique(ele[:,-1])
            for i in range(len(labels)):
                tmp = ele[ele[:,-1]==labels[i], :-1]
                idx = np.int32(np.unique(tmp) - 1)
                self.region[idx] = labels[i]
            self.elements = np.ascontiguousarray(np.sort(ele[:,:-1],axis=1), dtype=np.float64)
        else:
            raise ValueError('Error: elements in wrong format')

        # find the boundary nodes: find faces that are referred to only once
        faces = np.r_[ele[:, [0,1,2]], 
                      ele[:, [0,1,3]],
                      ele[:, [0,2,3]],
                      ele[:, [1,2,3]]]
        
        faces = np.sort(faces)
        unique_faces, cnt = np.unique(faces, axis=0, return_counts=1)
        bnd_faces = unique_faces[cnt==1, :]
        bndvtx = np.unique(bnd_faces)
        self.bndvtx = np.zeros(nodes.shape[0])
        self.bndvtx[np.int32(bndvtx-1)] = 1
        # area and support for each element
        self.element_area = utils.cpulib.ele_area(self.nodes, self.elements)
        self.support = utils.cpulib.mesh_support(self.nodes, self.elements, self.element_area)
        self.dimension = 3
        if np.any(prop != None):
            self.set_prop(prop)
        else:
            print('Warning: optical properties not specified')
        if src != None:
            self.source = copy.deepcopy(src)
            self.source.touch_sources(self)
        else:
            print('Warning: no sources specified')
        if det != None:
            self.meas = copy.deepcopy(det)
            self.meas.touch_detectors(self)
        else:
            print('Warning: no detectors specified')
        if np.all(link != None):
            if link.shape[1]==3:
                self.link = copy.deepcopy(np.ascontiguousarray(link, dtype=np.int32))
            elif link.shape[1]==2:
                self.link = copy.deepcopy(np.ascontiguousarray(np.c_[link, np.ones(link.shape[0])], dtype=np.int32))
            else:
                print('Warning: link in wrong format. Ignored.')
        else:
            print('Warning: no link specified')
    
    def from_volume(self, vol, param = utils.MeshingParams(), prop = None, src = None, det = None, link = None):
        """
        Construct mesh from a segmented 3D volume using the built-in CGAL mesher. Calls fluormesh.from_solid after meshing step.

        Parameters
        ----------
        vol : uint8 NumPy array
            3D segmented volume to be meshed. 0 is considered as outside. Regions labeled using unique integers.
        param : nirfasterff.utils.MeshingParams, optional
            parameters used in the CGAL mesher. If not specified, uses the default parameters defined in nirfasterff.utils.MeshingParams().
            
            Please modify fields xPixelSpacing, yPixelSpacing, and SliceThickness if your volume doesn't have [1,1,1] resolution
            
            See :func:`~nirfasterff.utils.MeshingParams()` for details.
        prop : double NumPy array, optional
            If not `None`, calls `fluormesh.set_prop()` and sets the optical properties in the mesh. The default is None.
            
            See :func:`~nirfasterff.base.fluor_mesh.fluormesh.set_prop()` for details. 
        src : nirfasterff.base.optode, optional
            If not `None`, sets the sources and moves them to the appropriate locations. The default is None.
            
            See :func:`~nirfasterff.base.optodes.optode.touch_sources()` for details.
        det : nirfasterff.base.optode, optional
            If not `None`, sets the detectors and moves them to the appropriate locations. The default is None.
            
            See :func:`~nirfasterff.base.optodes.optode.touch_detectors()` for details.
        link : int32 NumPy array, optional
            If not `None`, sets the channel information. Uses one-based indexing. The default is None.
            
            Each row represents a channel, in the form of `[src, det, active]`, where `active` is 0 or 1
            
            If `link` contains only two columns, all channels are considered active.

        Returns
        -------
        None.

        """
        if len(vol.shape) != 3:
            raise TypeError('Error: vol should be a 3D matrix in unit8')
        print('Running CGAL mesher', flush=1)
        ele, nodes = meshing.RunCGALMeshGenerator(vol, param)
        print('Converting to NIRFAST format', flush=1)
        self.from_solid(ele, nodes, prop, src, det, link)
    
    def set_prop(self, prop):
        """
        Set optical properties of the whole mesh, using information provided in prop.

        Parameters
        ----------
        prop : double NumPy array
            optical property info, similar to the MCX format::
                
                [region muax(mm-1) musxp(mm-1) ri muam(mm-1) musmp(mm-1) muaf(mm-1) eta tau]
                [region muax(mm-1) musxp(mm-1) ri muam(mm-1) musmp(mm-1) muaf(mm-1) eta tau]
                [...]
                
            where 'region' is the region label, and they should match exactly with unique(mesh.region). The order doesn't matter.

        Returns
        -------
        None.

        """
       
        num_nodes = self.nodes.shape[0]
        self.muax = np.zeros(num_nodes)
        self.musx = np.zeros(num_nodes)
        self.kappax = np.zeros(num_nodes)
        self.muam = np.zeros(num_nodes)
        self.musm = np.zeros(num_nodes)
        self.kappam = np.zeros(num_nodes)
        self.muaf = np.zeros(num_nodes)
        self.eta = np.zeros(num_nodes)
        self.tau = np.zeros(num_nodes)
        self.c = np.zeros(num_nodes)
        self.ksi = np.zeros(num_nodes)
        self.ri = np.zeros(num_nodes)
        
        if prop.shape[0]!=len(np.unique(self.region)) or (prop.shape[0]==len(np.unique(self.region)) and np.any(np.sort(prop[:,0])-np.unique(self.region))):
            print('Warning: regions in mesh and regions in prop matrix mismatch. Ignored.')
        elif prop.shape[1]!=9:
            print('Warning: prop matrix has wrong number of columns. Should be: region muax(mm-1) musxp(mm-1) ri muam(mm-1) musmp(mm-1) muaf(mm-1) eta tau. Ignored')
        else:
            labels = prop[:,0]
            for i in range(len(labels)):
                self.muax[self.region==labels[i]] = prop[i,1]
                self.musx[self.region==labels[i]] = prop[i,2]
                self.ri[self.region==labels[i]] = prop[i,3]
                self.muam[self.region==labels[i]] = prop[i,4]
                self.musm[self.region==labels[i]] = prop[i,5]
                self.muaf[self.region==labels[i]] = prop[i,6]
                self.eta[self.region==labels[i]] = prop[i,7]
                self.tau[self.region==labels[i]] = prop[i,8]
            self.kappax = 1.0/(3.0*(self.muax + self.musx))
            self.kappam = 1.0/(3.0*(self.muam + self.musm))
            c0 = 299792458000.0 # mm/s
            self.c = c0/self.ri
            n_air = 1.
            n = self.ri/n_air
            R0 = ((n-1.)**2)/((n+1.)**2)
            theta = np.arcsin(1.0/n)
            A = (2.0/(1.0 - R0) -1. + np.abs(np.cos(theta))**3) / (1.0 - np.abs(np.cos(theta))**2)
            self.ksi = 1.0 / (2*A)
            
    def change_prop(self, idx, prop):
        """
        Change optical properties (muax, musxp, ri, muam, musmp, muaf, eta, and tau) at nodes specified in idx, and automatically change fields kappax, kappam, c, and ksi as well

        Parameters
        ----------
        idx : list or NumPy array or -1
            zero-based indices of nodes to change. If `idx==-1`, function changes all the nodes
        prop : list or NumPy array of length 8
            new optical properties to be assigned to the specified nodes. [muax(mm-1) musxp(mm-1) ri muam(mm-1) musmp(mm-1) muaf(mm-1) eta tau(sec)].

        Returns
        -------
        None.

        """
       
        if np.size(idx)==1 and idx==-1:
            idx = np.arange(self.nodes.shape[0])
        idx = np.array(idx, dtype = np.int32)
        self.muax[idx] = prop[0]
        self.musx[idx] = prop[1]
        self.ri[idx] = prop[2]
        self.muam[idx] = prop[3]
        self.musm[idx] = prop[4]
        self.muaf[idx] = prop[5]
        self.eta[idx] = prop[6]
        self.tau[idx] = prop[7]
        
        self.kappax = 1.0 / (3.0*(self.muax + self.musx))
        self.kappam = 1.0 / (3.0*(self.muam + self.musm))
        c0 = 299792458000.0 # mm/s
        self.c = c0/self.ri
        n_air = 1.
        n = self.ri/n_air
        R0 = ((n-1.)**2)/((n+1.)**2)
        theta = np.arcsin(1.0/n)
        A = (2.0/(1.0 - R0) -1. + np.abs(np.cos(theta))**3) / (1.0 - np.abs(np.cos(theta))**2)
        self.ksi = 1.0 / (2*A)
    
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
    
    def save_nirfast(self, filename):
        """
        Save mesh in the classic NIRFASTer ASCII format, which is directly compatible with the Matlab version

        Parameters
        ----------
        filename : str
            name of the file to be saved as. Should have no extensions.

        Returns
        -------
        None.

        """
        # save nodes
        np.savetxt(filename+'.node', np.c_[self.bndvtx, self.nodes], fmt='%.16g', delimiter='\t')
        # save elements
        np.savetxt(filename+'.elem', self.elements, fmt='%g', delimiter='\t')
        # save params
        kappax = 1.0/(3.0*(self.muax + self.musx))
        kappam = 1.0/(3.0*(self.muam + self.musm))
        np.savetxt(filename+'.param', np.c_[self.muax, kappax, self.ri, self.muam, kappam, self.muaf, self.eta, self.tau], fmt='%g',
                   delimiter=' ', header='fluor', comments='')
        # save regions
        np.savetxt(filename+'.region', self.region, fmt='%g', delimiter='\t')
        # save sources, if exist
        if len(self.source.coord)==0:
            if os.path.isfile(filename+'.source'):
                os.remove(filename+'.source')
        else:
            if len(self.source.int_func)>0 and self.source.int_func.shape[0]==self.source.coord.shape[0]:
                if self.dimension==2:
                    hdr = 'num x y ele ip1 ip2 ip3'
                elif self.dimension==3:
                    hdr = 'num x y z ele ip1 ip2 ip3 ip4'
                data = np.c_[self.source.num, self.source.coord, self.source.int_func]
            else:
                if self.dimension==2:
                    hdr = 'num x y'
                elif self.dimension==3:
                    hdr = 'num x y z'
                data = np.c_[self.source.num, self.source.coord]
            if self.source.fixed == 1:
                hdr = 'fixed\n' + hdr
            np.savetxt(filename+'.source', data, fmt='%.16g', delimiter=' ', header=hdr, comments='')
        # save detectors, if exist
        if len(self.meas.coord)==0:
            if os.path.isfile(filename+'.meas'):
                os.remove(filename+'.meas')
        else:
            if len(self.meas.int_func)>0 and self.meas.int_func.shape[0]==self.meas.coord.shape[0]:
                if self.dimension==2:
                    hdr = 'num x y ele ip1 ip2 ip3'
                elif self.dimension==3:
                    hdr = 'num x y z ele ip1 ip2 ip3 ip4'
                data = np.c_[self.meas.num, self.meas.coord, self.meas.int_func]
            else:
                if self.dimension==2:
                    hdr = 'num x y'
                elif self.dimension==3:
                    hdr = 'num x y z'
                data = np.c_[self.meas.num, self.meas.coord]
            if self.meas.fixed == 1:
                hdr = 'fixed\n' + hdr
            np.savetxt(filename+'.meas', data, fmt='%.16g', delimiter=' ', header=hdr, comments='')
        # save link, if exist
        if len(self.link)==0:
            if os.path.isfile(filename+'.link'):
                os.remove(filename+'.link')
        else:
            hdr = 'source detector active'
            np.savetxt(filename+'.link', self.link, fmt='%g', delimiter=' ', header=hdr, comments='')
    
    def femdata(self, freq, solver=utils.get_solver(), opt=utils.SolverOptions(), xflag=True, mmflag=True, flflag=True):
        """
        Calculates fluences for each source using a FEM solver, and then the boudary measurables for each channel 
        
        If `mesh.vol` is set, fluence data will be represented in volumetric space
        
        See :func:`~nirfasterff.forward.femdata.femdata_fl_CW()` and :func:`~nirfasterff.forward.femdata.femdata_fl_FD()` for details

        Parameters
        ----------
        freq : double
            modulation frequency in Hz. If CW, set to zero and a more efficient CW solver will be used.
        solver : str, optional
            Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
        opt : nirfasterff.utils.SolverOptions, optional
            Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
            
            See :func:`~nirfasterff.utils.SolverOptions` for details
        xflag : bool, optional
            if intrinsic excitation field is calculated. The default is True.
        mmflag : bool, optional
            if intrinsic emission field is calculated. The default is True.
        flflag : bool, optional
            if fluorescence field is calculated. If set True, xflag must also be True. The default is True.

        Returns
        -------
        data : nirfasterff.base.FLdata
            fluence and boundary measurables given the mesh and optodes.
            
            See :func:`~nirfasterff.base.data.FLdata` for details.
        infox : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, for intrinsic excitation.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details
        infom : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, for intrinsic emission.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details
        infofl : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, for fluorescence emission.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details

        """
       
        # freq in Hz
        if freq==0:
            data, infox, infom, infofl = forward.femdata_fl_CW(self, solver, opt, xflag, mmflag, flflag)
            return data, infox, infom, infofl
        else:
            data, infox, infom, infofl = forward.femdata_fl_FD(self, freq, solver, opt, xflag, mmflag, flflag)
            return data, infox, infom, infofl
    
    def femdata_tpsf(self, tmax, dt, savefield=False, beautify=True, solver=utils.get_solver(), opt=utils.SolverOptions()):
        """
        Calculates TPSF at each location in the mesh for each source using a FEM solver, and then the boudary measurables for each channel 
        
        This is done for both excitation and fluorescence emission
        
        If `mesh.vol` is set and `savefield` is set to `True`, internal TPSF data will be represented in volumetric space
        
        See :func:`~nirfasterff.forward.femdata.femdata_fl_TR()` for details

        Parameters
        ----------
        tmax : double
            maximum time simulated, in seconds.
        dt : double
            size of each time step, in seconds.
        savefield : bool, optional
            If True, the internal TPSFs are also returned. If False, only boundary TPSFs are returned and data.phix and data.phifl will be empty. 
            
            The default is False.
        beautify : bool, optional
            If true, zeros the initial unstable parts of the boundary TPSFs. The default is True.
        solver : str, optional
            Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
        opt : nirfasterff.utils.SolverOptions, optional
            Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
            
            See :func:`~nirfasterff.utils.SolverOptions` for details

        Returns
        -------
        data : nirfasterff.base.flTPSFdata
            internal and boundary TPSFs given the mesh and optodes, both excitation and fluorescence emission.
            
            See :func:`~nirfasterff.base.data.flTPSFdata` for details.
        infox : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, excitation. 
            
            Only the convergence info of the last time step is returned.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details
        infom : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, fluorescence emission. 
            
            Only the convergence info of the last time step is returned.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details

        """
        data, infox, infom = forward.femdata_fl_TR(self, tmax, dt, savefield, beautify, solver, opt)
        return data, infox, infom
    
    def femdata_moments(self, max_moments=3, savefield=False, solver=utils.get_solver(), opt=utils.SolverOptions()):
        """
        Calculates TR moments at each location in the mesh for each source directly using Mellin transform, and then the boudary measurables for each channel 
        
        This is done for both excitation and fluorescence emission
        
        If `mesh.vol` is set and `savefield` is set to `True`, internal moments data will be represented in volumetric space
        
        See :func:`~nirfasterff.forward.femdata.femdata_fl_TR_moments()` for details

        Parameters
        ----------
        max_moments : int32, optional
            max order of moments to calculate. That is, 0th, 1st, 2nd, .., max_moments-th will be calculated. The default is 3.
        savefield : bool, optional
            If True, the internal moments are also returned. If False, only boundary moments are returned and data.phix and data.phifl will be empty. 
            
            The default is False.
        solver : str, optional
            Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically determined (GPU prioritized) if not specified
        opt : nirfasterff.utils.SolverOptions, optional
            Solver options. Uses default parameters if not specified, and they should suffice in most cases. 
            
            See :func:`~nirfasterff.utils.SolverOptions` for details

        Returns
        -------
        data : nirfasterff.base.flTRMomentsdata
            internal and boundary moments given the mesh and optodes, both excitation and fluorescence emission.
            
            See :func:`~nirfasterff.base.femdata.flTRMomentsdata` for details.
        infox : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, excitation. 
            
            Only the convergence info of highest order moments is returned.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details
        infom : nirfasterff.utils.ConvergenceInfo
            convergence information of the solver, fluorescence emission. 
            
            Only the convergence info of highest order moments is returned.
            
            See :func:`~nirfasterff.utils.ConvergenceInfo` for details

        """
        data, infox, infom = forward.femdata_fl_TR_moments(self, max_moments, savefield, solver, opt)
        return data, infox, infom
    
    def jacobian(self, freq=0, normalize=True, solver = utils.get_solver(), opt = utils.SolverOptions()):
        """
        Calculates the Jacobian matrix
        
        Returns the Jacobian, direct field data, and the adjoint data
        
        Structure of the Jacobian is detailed in :func:`~nirfasterff.inverse.jacobian_fl_CW()` and :func:`~nirfasterff.inverse.jacobian_fl_FD()`

        Parameters
        ----------
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
        ValueError
            if modulation frequency is negative.

        Returns
        -------
        J : double NumPy array
            The Jacobian matrix. Size (NChannel, NVoxel*2) or (NChannel, NVoxel) 
        data1 : nirfasterff.base.FLdata
            The calculated direct field. The same as directly calling mesh.femdata(freq)
        data2 : nirfasterff.base.FLdata
            The calculated adjoint field. The same as calling mesh.femdata(freq) AFTER swapping the locations of sources and detectors

        """
        if freq>=0:
            J, data1, data2 = inverse.jacobian_fl_FD(self, freq, normalize, solver, opt)
        else:
            raise ValueError('Modulation frequency must be non-negative')
        return J, data1, data2
        
    def isvol(self):
        """
        Check if convertion matrices between mesh and volumetric spaces are calculated

        Returns
        -------
        bool
            True if attribute `.vol` is calculate, False if not.

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
        
        
        self.vol.gridinmesh, self.vol.meshingrid, self.vol.mesh2grid, self.vol.grid2mesh = utils.gen_intmat_impl(self, xgrid, ygrid, zgrid)
        