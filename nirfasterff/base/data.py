"""
Defining some data classes, which are the return types of the fem data calculation functions
"""

import numpy as np
import copy
from scipy import sparse

class meshvol:
    """
    Small class holding the information needed for converting between mesh and volumetric space. Values calculated by nirfasterff.base.*mesh.gen_intmat
    
    Note that the volumetric space, defined by xgrid, ygrid, and zgrid (empty for 2D mesh), must be uniform
    
    Attributes
    ----------
        xgrid: double Numpy array
            x grid of the volumetric space. In mm
        ygrid: double Numpy array
            y grid of the volumetric space. In mm
        zgrid: double Numpy array
            z grid of the volumetric space. In mm. Empty for 2D meshes
        mesh2grid: double CSR sparse matrix
            matrix converting a vector in mesh space to volumetric space, done by mesh2grid.dot(data)
            
            The result is vectorized in 'F' (Matlab) order
            
            Size: (len(xgrid)*len(ygrid)*len(zgrid), NNodes)
        gridinmesh: int32 NumPy array
            indices (one-based) of data points in the volumetric space that are within the mesh space, vectorized in 'F' order.
        res: double NumPy array
            resolution in x, y, z (if 3D) direction, in mm. Size (2,) or (3,)
        grid2mesh: double CSR sparse matrix
            matrix converting volumetric data, vectorized in 'F' order, to mesh space. Done by grid2mesh.dot(data)
            
            Size (Nnodes, len(xgrid)*len(ygrid)*len(ygrid))
        meshingrid: int32 NumPy array
            indices (one-based) of data points in the mesh space that are within the volumetric space
        
    """
    def __init__(self):
        self.xgrid = []
        self.ygrid = []
        self.zgrid = []
        self.mesh2grid = sparse.csc_matrix([])
        self.gridinmesh = []
        self.res = []
        self.grid2mesh = sparse.csc_matrix([])
        self.meshingrid = []

class FDdata:
    """
    Class holding FD/CW data.
    
    Attributes
    ----------
    phi: double Numpy array
        Fluence from each source. If mesh contains non-tempty field vol, this will be represented on the grid. Last dimension has the size of the number of sources
    complex: double or complex double Numpy vector
        Complex amplitude of each channel. Same as amplitude in case of CW data
    link: int32 NumPy array
        Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
    amplitude: double Numpy vector
        Absolute amplitude of each channel. I.e. amplitude=abs(complex)
    phase: double Numpy vector
        phase data of each channel. All zero in case of CW data
    vol: nirfaseterff.base.meshvol
        Information needed to convert between volumetric and mesh space. Copied from mesh.vol
    """
    def __init__(self):
        self.phi = []
        self.complex = []
        self.link = []
        self.amplitude = []
        self.phase = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the field phi

        Parameters
        ----------
        mesh : nirfasterff.base.stndmesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if len(mesh.vol.xgrid)>0:
            if len(self.vol.xgrid)>0:
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                phi_mesh = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]), order='F'))
                if len(self.vol.zgrid)>0:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                else:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
            else:
                if len(mesh.vol.zgrid)>0:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                else:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
            
            self.phi = tmp
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty, the function does nothing.
        
        CAUTION: This OVERRIDES the field phi

        Parameters
        ----------
        mesh : nirfasterff.base.stndmesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            self.phi = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

class FLdata:
    """
    Class holding FD/CW fluorescence data.
    
    Attributes
    ----------
        phix: double Numpy array
            intrinsic fluence from each source at excitation wavelength. If mesh contains non-tempty field vol, this will be represented on the grid. Last dimension has the size of the number of sources
        phimm: double Numpy array 
            intrinsic from each source at emission wavelength.
        phifl: double Numpy array 
            fluorescence emission fluence
        complexx: double or complex double Numpy vector
            Complex amplitude of each channel, intrinsic excitation. Same as amplitude in case of CW data
        complexmm: double or complex double Numpy vector
            Complex amplitude of each channel, intrinsic emission. Same as amplitude in case of CW data
        complexfl: double or complex double Numpy vector
            Complex amplitude of each channel, fluorescence emission. Same as amplitude in case of CW data
        link: int32 NumPy array
            Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
        amplitudex: double Numpy vector
            Absolute amplitude of each channel, intrinsic excitation. I.e. amplitudex=abs(complexx)
        amplitudemm: double Numpy vector
            Absolute amplitude of each channel, intrinsic emission. I.e. amplitudemm=abs(complexmm)
        amplitudefl: double Numpy vector
            Absolute amplitude of each channel, fluorescence emission. I.e. amplitudefl=abs(complexfl)
        phasex: double Numpy vector
            phase data of each channel, intrinsic excitation. All zero in case of CW data
        phasemm: double Numpy vector
            phase data of each channel, intrinsic emission. All zero in case of CW data
        phasefl: double Numpy vector
            phase data of each channel, fluorescence emission. All zero in case of CW data
        vol: nirfaseterff.base.meshvol
            Information needed to convert between volumetric and mesh space. Copied from mesh.vol
    """
    def __init__(self):
        self.phix = []
        self.phimm = []
        self.phifl = []
        self.complexx = []
        self.complexmm = []
        self.complexfl = []
        self.link = []
        self.amplitudex = []
        self.amplitudemm = []
        self.amplitudefl = []
        self.phasex = []
        self.phasemm = []
        self.phasefl = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the fields phix, phimm, and phifl, if they are defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        
        xflag = len(self.phix)>0
        mmflag = len(self.phimm)>0
        flflag = len(self.phifl)>0
        tmpx = []
        tmpmm = []
        tmpfl = []
        
        if len(mesh.vol.xgrid)>0:
            if len(self.vol.xgrid)>0:
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                if xflag:
                    phix_mesh = self.vol.grid2mesh.dot(np.reshape(self.phix, (-1, self.phix.shape[-1]), order='F'))
                if mmflag:
                    phimm_mesh = self.vol.grid2mesh.dot(np.reshape(self.phimm, (-1, self.phimm.shape[-1]), order='F'))
                if flflag:
                    phifl_mesh = self.vol.grid2mesh.dot(np.reshape(self.phifl, (-1, self.phifl.shape[-1]), order='F'))
                    
                if len(self.vol.zgrid)>0:
                    if xflag:
                        tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    if mmflag:
                        tmpmm = np.reshape(mesh.vol.mesh2grid.dot(phimm_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    if flflag:
                        tmpfl = np.reshape(mesh.vol.mesh2grid.dot(phifl_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                else:
                    if xflag:
                        tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                    if mmflag:
                        tmpmm = np.reshape(mesh.vol.mesh2grid.dot(phimm_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                    if flflag:
                        tmpfl = np.reshape(mesh.vol.mesh2grid.dot(phifl_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
            else:
                if len(mesh.vol.zgrid)>0:
                    if xflag:
                        tmpx = np.reshape(mesh.vol.mesh2grid.dot(self.phix), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    if mmflag:
                        tmpmm = np.reshape(mesh.vol.mesh2grid.dot(self.phimm), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    if flflag:
                        tmpfl = np.reshape(mesh.vol.mesh2grid.dot(self.phifl), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                else:
                    if xflag:
                        tmpx = np.reshape(mesh.vol.mesh2grid.dot(self.phix), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                    if mmflag:
                        tmpmm = np.reshape(mesh.vol.mesh2grid.dot(self.phimm), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                    if flflag:
                        tmpfl = np.reshape(mesh.vol.mesh2grid.dot(self.phifl), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
            
            self.phix = tmpx
            self.phimm = tmpmm
            self.phifl = tmpfl
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty, the function does nothing.
        
        CAUTION: This OVERRIDES fields phix, phimm, and phifl, if they are defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            xflag = len(self.phix)>0
            mmflag = len(self.phimm)>0
            flflag = len(self.phifl)>0
            if xflag:
                self.phix = self.vol.grid2mesh.dot(np.reshape(self.phix, (-1, self.phix.shape[-1]), order='F'))
            if mmflag:
                self.phimm = self.vol.grid2mesh.dot(np.reshape(self.phimm, (-1, self.phimm.shape[-1]), order='F'))
            if flflag:
                self.phifl = self.vol.grid2mesh.dot(np.reshape(self.phifl, (-1, self.phifl.shape[-1]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

class TRMomentsdata:
    """
    Class holding time-resolved moments data calculated using Mellin transform.
    
    Attributes
    ----------
        phi: double Numpy array or None
            moments from each source at each spatial location. If mesh contains non-tempty field vol, this will be represented on the grid
            
            Shape: NNodes x num_sources x (max_moment_order + 1)
            
            OR: len(xgrid) x len(ygrid) x len(zgrid) x num_sources x (max_moment_order + 1)
            
            None by default, and only contains data if 'field' option is set to True when calculating forward data.
        moments: double Numpy vector
            moments for each channel. i-th column contains i-th moment as measured at each channel. Size: (NChannels, max_moment_order + 1)
        link: int32 NumPy array
            Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
        vol: nirfaseterff.base.meshvol
            Information needed to convert between volumetric and mesh space. Copied from mesh.vol

    """
    def __init__(self):
        self.phi = None
        self.link = []
        self.moments = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty or data.phi==None, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the field phi, if it is defined

        Parameters
        ----------
        mesh : nirfasterff.base.stndmesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phi==None):
            return
        
        if mesh.isvol():
            if self.isvol():
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                phi_mesh = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]*self.phi.shape[-2]), order='F'))
                if len(self.vol.zgrid)>0:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            else:
                if len(mesh.vol.zgrid)>0:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            
            self.phi = tmp
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty or data.phi==None, the function does nothing.
        
        CAUTION: This OVERRIDES field phi, if it is defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phi==None):
            return
        
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            self.phi = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]*self.phi.shape[-2]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

class TPSFdata:
    """
    Class holding time-resolved TPSF data.
    
    Attributes
    ----------
        phi: double NumPy array or None
            TPSF from each source at each spatial location. If mesh contains non-tempty field vol, this will be represented on the grid
            
            Shape: NNodes x num_sources x time_steps
            
            OR: len(xgrid) x len(ygrid) x len(zgrid) x num_sources x time_steps
            
            None by default, and only contains data if 'field' option is set to True when calculating forward data.
        time: double NumPy vector
            time vector, in seconds
        tpsf: double NumPy array
            TPSF measured at each channel. Size: (NChannels, time_steps)
        link: int32 NumPy array
            Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
        vol: nirfaseterff.base.meshvol
            Information needed to convert between volumetric and mesh space. Copied from mesh.vol

    """
    def __init__(self):
        self.phi = None
        self.time = []
        self.link = []
        self.tpsf = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty or data.phi==None, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the field phi, if it is defined

        Parameters
        ----------
        mesh : nirfasterff.base.stndmesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phi==None):
            return
        
        if mesh.isvol():
            if self.isvol():
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                phi_mesh = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]*self.phi.shape[-2]), order='F'))
                if len(self.vol.zgrid)>0:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            else:
                phi_tmp = np.reshape(self.phi, (mesh.nodes.shape[0], -1))
                if len(mesh.vol.zgrid)>0:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_tmp), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmp = np.reshape(mesh.vol.mesh2grid.dot(phi_tmp), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            
            self.phi = tmp
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty or data.phi==None, the function does nothing.
        
        CAUTION: This OVERRIDES field phi, if it is defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phi==None):
            return
        
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            self.phi = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]*self.phi.shape[-2]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

class flTRMomentsdata:
    """
    Class holding fluorescence TR moments data calculated using Mellin transform.
    
    Attributes
    ----------
        phix: double NumPy array
            moments at excitation wavelength from each source at each spatial location. If mesh contains non-tempty field vol, this will be represented on the grid
            
            Shape: NNodes x num_sources x (max_moment_order + 1)
            
            OR: len(xgrid) x len(ygrid) x len(zgrid) x num_sources x (max_moment_order + 1)
            
            None by default, and only contains data if 'field' option is set to True when calculating forward data.
        phifl: double NumPy array
            similar to phix, but for fluorescence emission
        momentsx: double NumPy array
            moments for each channel, exciation. i-th column contains i-th moment. Size: (NChannels, max_moment_order + 1)
        momentsfl: double NumPy array
            moments for each channel, fluorescence emission. i-th column contains i-th moment. Size: (NChannels, max_moment_order + 1)
        link: int32 NumPy array
            Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
        vol: nirfaseterff.base.meshvol
            Information needed to convert between volumetric and mesh space. Copied from mesh.vol

    """
    def __init__(self):
        self.phix = None
        self.phifl = None
        self.link = []
        self.momentsx = []
        self.momentsfl = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty or data.phix==None, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the fields phix and phifl, if they are defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phix==None):
            return
        
        if mesh.isvol():
            if self.isvol():
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                phix_mesh = self.vol.grid2mesh.dot(np.reshape(self.phix, (-1, self.phix.shape[-1]*self.phix.shape[-2]), order='F'))
                phifl_mesh = self.vol.grid2mesh.dot(np.reshape(self.phifl, (-1, self.phifl.shape[-1]*self.phifl.shape[-2]), order='F'))
                if len(self.vol.zgrid)>0:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(phifl_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(phifl_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            else:
                if len(mesh.vol.zgrid)>0:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(self.phix), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(self.phifl), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(self.phix), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(self.phifl), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            
            self.phix = tmpx
            self.phifl = tmpm
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty or data.phix==None, the function does nothing.
        
        CAUTION: This OVERRIDES fields phix and phifl, if they are defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phix==None):
            return
        
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            self.phix = self.vol.grid2mesh.dot(np.reshape(self.phix, (-1, self.phix.shape[-1]*self.phix.shape[-2]), order='F'))
            self.phifl = self.vol.grid2mesh.dot(np.reshape(self.phifl, (-1, self.phifl.shape[-1]*self.phifl.shape[-2]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

class flTPSFdata:
    """
    Class holding fluorescence time-resolved TPSF data.
    
    Attributes
    ----------
        phix: double NumPy array
            TPSF at exciation wavelength from each source at each spatial location. If mesh contains non-tempty field vol, this will be represented on the grid
            
            Shape: NNodes x num_sources x time_steps
            
            OR: len(xgrid) x len(ygrid) x len(zgrid) x num_sources x time_steps
            
            None by default, and only contains data if 'field' option is set to True when calculating forward data.
        phifl: double NumPy array
            similar to phix, but for fluorescence emission
        time: double NumPy vector
            time vector, in seconds
        tpsfx: double NumPy array
            TPSF measured at each channel, excitation. Size: (NChannels, time_steps)
        link: int32 NumPy array
            Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
        vol: nirfaseterff.base.meshvol
            Information needed to convert between volumetric and mesh space. Copied from mesh.vol
    
    """
    def __init__(self):
        self.phix = None
        self.phifl = None
        self.time = []
        self.link = []
        self.tpsfx = []
        self.tpsffl = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty or data.phix==None, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the fields phix and phifl, if they are defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phix==None):
            return
        
        if mesh.isvol():
            if self.isvol():
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                phix_mesh = self.vol.grid2mesh.dot(np.reshape(self.phix, (-1, self.phix.shape[-1]*self.phix.shape[-2]), order='F'))
                phifl_mesh = self.vol.grid2mesh.dot(np.reshape(self.phifl, (-1, self.phifl.shape[-1]*self.phifl.shape[-2]), order='F'))
                if len(self.vol.zgrid)>0:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(phifl_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(phifl_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            else:
                phix_tmp = np.reshape(self.phi, (mesh.nodes.shape[0], -1))
                phi,_tmp = np.reshape(self.phi, (mesh.nodes.shape[0], -1))
                if len(mesh.vol.zgrid)>0:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_tmp), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(tmpm), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmpx = np.reshape(mesh.vol.mesh2grid.dot(phix_tmp), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
                    tmpm = np.reshape(mesh.vol.mesh2grid.dot(tmpm), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            
            self.phix = tmpx
            self.phifl = tmpm
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty or data.phix==None, the function does nothing.
        
        CAUTION: This OVERRIDES fields phix and phifl, if they are defined

        Parameters
        ----------
        mesh : nirfasterff.base.fluormesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if np.all(self.phix==None):
            return
        
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            self.phix = self.vol.grid2mesh.dot(np.reshape(self.phix, (-1, self.phix.shape[-1]*self.phix.shape[-2]), order='F'))
            self.phifl = self.vol.grid2mesh.dot(np.reshape(self.phifl, (-1, self.phifl.shape[-1]*self.phifl.shape[-2]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

class DCSdata:
    """
    Class holding DCS data.
    
    Attributes
    ----------
        phi: double NumPy array
            steady-state fluence from each source. If mesh contains non-tempty field vol, this will be represented on the grid. 
            
            Last dimension has the size of the number of sources
            
            This is the same as nirfasterff.base.FDdata.phi, when modulation frequency is zero
            
        link: int32 NumPy array
            Defining all the channels (i.e. source-detector pairs). Copied from mesh.link
        amplitude: double NumPy vector
            Steady-state amplitude of each channel. Size (NChannel,)
            
            This is the same as nirfasterff.base.FDdata.amplitude, when modulation frequency is zero
        tau_DCS: double NumPy vector
            time vector in seconds
        phi_DCS: double NumPy array
            G1 in medium from each source at each time step . If mesh contains non-tempty field vol, this will be represented on the grid
            
            shape[-1] equals length of tau_DCS, and shape[-2] equals number of sources
        G1_DCS: double NumPy array
            G1 curve as is calculated from the correlation diffusion equation. Size: (NChannel, NTime)
        g1_DCS: double NumPy array
            g1 curve, i.e. G1 normalized by amplitudes. Size: (NChannel, NTime)
        vol: nirfaseterff.base.meshvol
            Information needed to convert between volumetric and mesh space. Copied from mesh.vol
    
    """
    def __init__(self):
        self.phi = []
        self.link = []
        self.amplitude = []
        self.tau_DCS = []
        self.phi_DCS = []
        self.G1_DCS = []
        self.g1_DCS = []
        self.vol = meshvol()
        
    def togrid(self, mesh):
        """
        Convert data to volumetric space as is defined in mesh.vol. If it is empty, the function does nothing.
        
        If data is already in volumetric space, function casts data to the new volumetric space
        
        CAUTION: This OVERRIDES the fields phi and phi_DCS

        Parameters
        ----------
        mesh : nirfasterff.base.dcsmesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if len(mesh.vol.xgrid)>0:
            if len(self.vol.xgrid)>0:
                print('Warning: data already in volumetric space. Recasted to the new volume.')
                phi_mesh = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]), order='F'))
                phiDCS_mesh = self.vol.grid2mesh.dot(np.reshape(self.phi_DCS, (-1, self.phi_DCS.shape[-1]*self.phi_DCS.shape[-2]), order='F'))
                if len(self.vol.zgrid)>0:
                    tmp1 = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    tmp2 = np.reshape(mesh.vol.mesh2grid.dot(phiDCS_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmp1 = np.reshape(mesh.vol.mesh2grid.dot(phi_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                    tmp2 = np.reshape(mesh.vol.mesh2grid.dot(phiDCS_mesh), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            else:
                if len(mesh.vol.zgrid)>0:
                    tmp1 = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, -1), order='F')
                    phi_DCS = np.reshape(self.phi_DCS, (self.phi_DCS.shape[0],-1), order='F')
                    tmp2 = np.reshape(mesh.vol.mesh2grid.dot(phi_DCS), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.vol.zgrid.size, mesh.source.num.size, -1), order='F')
                else:
                    tmp1 = np.reshape(mesh.vol.mesh2grid.dot(self.phi), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, -1), order='F')
                    phi_DCS = np.reshape(self.phi_DCS, (self.phi_DCS.shape[0],-1), order='F')
                    tmp2 = np.reshape(mesh.vol.mesh2grid.dot(phi_DCS), (mesh.vol.ygrid.size, mesh.vol.xgrid.size, mesh.source.num.size, -1), order='F')
            
            self.phi = tmp1
            self.phi_DCS = tmp2
            self.vol = copy.deepcopy(mesh.vol)
        else:
            print('Warning: no converting information found. Ignored. Please run mesh.gen_intmat() first.')
    
    def tomesh(self, mesh):
        """
        Convert data back to mesh space using information defined in mesh.vol. If data.vol is empty, the function does nothing.
        
        CAUTION: This OVERRIDES fields phi and phi_DCS

        Parameters
        ----------
        mesh : nirfasterff.base.dcsmesh
            mesh whose .vol attribute is used to do the conversion.

        Returns
        -------
        None.

        """
        if not self.isvol():
            print('Warning: data already in mesh space. Ignored.')
        else:
            self.phi = self.vol.grid2mesh.dot(np.reshape(self.phi, (-1, self.phi.shape[-1]), order='F'))
            self.phi_DCS = self.vol.grid2mesh.dot(np.reshape(self.phi_DCS, (-1, self.phi_DCS.shape[-1]*self.phi_DCS.shape[-2]), order='F'))
            self.vol = meshvol()
    
    def isvol(self):
        """
        Checks if data is in volumetric space.

        Returns
        -------
        bool
            True if data is in volumetric space, False if not.

        """
        if len(self.vol.xgrid):
            return True
        else:
            return False

