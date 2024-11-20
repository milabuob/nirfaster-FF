"""
Analytical solutions to the diffusion equation in semi-infinite media
"""
from nirfasterff import utils
from nirfasterff import math
from nirfasterff import base
import numpy as np
import copy
from scipy import special
from scipy import integrate

def semi_infinite_FD(mua, musp, n, freq, rho, z=0, boundary = 'exact', n_air = 1.0):
    """
    Calculates the frequency-domain fluence in space using the analytical solution to the diffusion equation in semi-infinite media.

    Parameters
    ----------
    mua : double
        absorption coefficient of the medium, in mm^-1.
    musp : double
        reduced scattering coefficient of the medium, in mm^-1.
    n : double
        refractive index of the medium.
    freq : double NumPy vector or scalar
        modulation frequency, in Hz.
    rho : double NumPy vector or scalar
        distance to the light source, projected to the x-y (i.e. boundary plane of the semi-infinite space) plane, in mm.
        
        Can be a vector, in which case fluences calculated at multiple locations will be returned
    z : double NumPy vector or scalar, optional
        depth of the location(s) of interest. 0 for boundary measurement.
        
        If a vector, it must have the same length as rho
        
        The default is 0.
    boundary : str, optional
        type of the boundary condition, which can be 'robin', 'approx', or 'exact'. The default is 'exact'.
        
        See :func:`~nirfasterff.utils.boundary_attenuation()` for details.
    n_air : double, optional
        refratcive index outside of the semi-infinite space, which is typically assumed to be air. The default is 1.0.

    Returns
    -------
    complex double NumPy array
        calculated complex fluence, where each row (or element in vector) corresponds to a location, as specified by rho and z, 
        
        and each column corresponds to a modulation frequency. Size (NLocation,), or (NLocation, NFreq)
        
    References
    -------
    Durduran et al, 2010, Rep. Prog. Phys. doi:10.1088/0034-4885/73/7/076701

    """
    z0 = 1./(mua + musp) # in mm
    c0 = 299792458000. # mm/s; light spead in vacuum
    cm = c0/n
    D = 1./(3. * (mua + musp)) # mm; diffusion coefficient
    
    omega = 2*np.pi*np.atleast_1d(freq)
    rho = np.array(1.0*rho)
    phi = np.zeros((np.size(rho), np.size(omega)), dtype=np.complex128)

    # get boundary attenuation
    A = utils.boundary_attenuation(n, n_air, boundary)
    
    r1 = np.sqrt((z - z0)**2 + rho**2)
    zb = 2.0 * z0 * A / 3.0
    rb = np.sqrt((z + 2*zb + z0)**2 + rho**2)
    
    for i in range(omega.size):
        k = np.sqrt((mua - 1j*omega[i]/cm) / D)
        phi[:,i] = (np.exp(-k*r1)/r1 - np.exp(-k*rb)/rb) / (4*np.pi*D)
    
    return phi.squeeze()

def semi_infinite_CW(mua, musp, n, rho, z=0, boundary = 'exact', n_air = 1.0):
    """
    Calculates the continuous-wave fluence in space using the analytical solution to the diffusion equation in semi-infinite media.
    
    Internaly calls the FD version with freq set to zero

    Parameters
    ----------
    mua : double
        absorption coefficient of the medium, in mm^-1.
    musp : double
        reduced scattering coefficient of the medium, in mm^-1.
    n : double
        refractive index of the medium.
    rho : double NumPy vector or scalar
        distance to the light source, projected to the x-y (i.e. boundary plane of the semi-infinite space) plane, in mm.
        
        Can be a vector, in which case fluences calculated at multiple locations will be returned
    z : double NumPy vector or scalar, optional
        depth of the location(s) of interest. 0 for boundary measurement.
        
        If a vector, it must have the same length as rho
        
        The default is 0.
    boundary : str, optional
        type of the boundary condition, which can be 'robin', 'approx', or 'exact'. The default is 'exact'.
        
        See :func:`~nirfasterff.utils.boundary_attenuation()` for details.
    n_air : double, optional
        refratcive index outside of the semi-infinite space, which is typically assumed to be air. The default is 1.0.

    Returns
    -------
    double NumPy vector
        calculated fluence, where each element corresponds to a location, as specified by rho and z. Size (NLocation,)
                
    References
    -------
    Durduran et al, 2010, Rep. Prog. Phys. doi:10.1088/0034-4885/73/7/076701

    """
    return np.real(semi_infinite_FD(mua, musp, n, 0., rho, z=0, boundary = 'exact', n_air = 1.0))

def semi_infinite_DCS(mua, musp, n, rho, aDb, wvlength, tvec, z=0, boundary = 'exact', n_air = 1.0, normalize=0):
    """
    Calculates DCS G1 curve using the analytical solution to the correlation diffusion equation in semi-infinite media
    
    Function assumes Brownian motion, that is, :math:`\\langle\\Delta r^2\\rangle=6\\alpha Db\\tau`

    Parameters
    ----------
    mua : double
        absorption coefficient of the medium, in mm^-1.
    musp : double
        reduced scattering coefficient of the medium, in mm^-1.
    n : double
        refractive index of the medium.
    rho : scalar
        distance to the light source, projected to the x-y (i.e. boundary plane of the semi-infinite space) plane, in mm.
    aDb : double
        The lumped flow parameter :math:`\\alpha Db` in Brownian motion.
    wvlength : double or int
        wavelength used, in nm.
    tvec : double Numpy vector
        time vector used to calculate the G1 curve. It is usually a good idea to use log space.
    z : double, optional
        depth of the location of interest. 0 for boundary measurement. The default is 0.
    boundary : str, optional
        type of the boundary condition, which can be 'robin', 'approx', or 'exact'. The default is 'exact'.
        
        See :func:`~nirfasterff.utils.boundary_attenuation()` for details.
    n_air : double, optional
        refratcive index outside of the semi-infinite space, which is typically assumed to be air. The default is 1.0.
    normalize : bool, optional
        if true, returns the normalized g1 curve, instead of G1. The default is 0.

    Returns
    -------
    double NumPy vector
        G1 or g1 curve calculated at the given location and time points.
    
    References
    -------
    Durduran et al, 2010, Rep. Prog. Phys. doi:10.1088/0034-4885/73/7/076701

    """
    k = 2*np.pi / (wvlength/1e6)
    mua2 = mua + 2*musp*k*k*aDb*tvec
    G1 = np.zeros(np.size(tvec))
    for i in range(np.size(tvec)):
        G1[i] = semi_infinite_CW(mua2[i], musp, n, rho, z, boundary, n_air)
    if normalize:
        amp = semi_infinite_CW(mua, musp, n, rho, 0., boundary, n_air)
        g1 = G1/amp
        return g1
    else:
        return G1

def semi_infinite_TR(mua, musp, n, rho, T, dt, z=0, boundary='EBC-Robin'):
    """
    Calculates TPSF at a given location using the analytical solution to the diffusion equation in semi-infinite media
    

    Parameters
    ----------
    mua : double
        absorption coefficient of the medium, in mm^-1.
    musp : double
        reduced scattering coefficient of the medium, in mm^-1.
    n : double
        refractive index of the medium.
    rho : scalar
        distance to the light source, projected to the x-y (i.e. boundary plane of the semi-infinite space) plane, in mm.
    T : double NumPy vector or scalar
        if a scalar, it is the total amount of time and time vector will be generated also based on dt (see below).
        
        if a vector, it is directly used as the time vector, and the argument dt will be ignored. Unit: seconds
    dt : double
        step size of the time vector (in seconds). If the argument T is a vector, it will be ignored.
    z : double, optional
        depth of the location of interest. 0 for boundary measurement. The default is 0.
        
        Note that when `z=0`, the function returns reflectance, instead of fluence. Please refer to the References for detail.
    boundary : str, optional
        type of the boundary condition, which can be (case insensitive),
        
        'PCB-exact' - partial current boundary condition, with exact internal reflectance
        
        'PCB-approx' - partial current boundary condition, with Groenhuis internal reflectance approximation
        
        'PCB-Robin' - partial current boundary condition, with internal reflectance derived from Fresnel's law
        
        'EBC-exact' - extrapolated boundary condition, with exact internal reflectance
        
        'EBC-approx' - extrapolated boundary condition, with Groenhuis internal reflectance approximation
        
        'EBC-Robin' - extrapolated boundary condition, with internal reflectance derived from Fresnel's law
        
        'ZBC' - zero boundary condition
        
        The default is 'EBC-Robin'.
        
        See :func:`~nirfasterff.utils.boundary_attenuation()` for the differences between 'exact', 'approx' and 'robin'

    Raises
    ------
    ValueError
        if boundary condition is not of a recognized kind.

    Returns
    -------
    phi : double Numpy array
        Coumn 0: the time vector; Column 1: calculated TPSF at the given location. Size (NTime, 2)
    
    References
    -------
    Hielscher et al., 1995, Phys. Med. Biol. doi:10.1088/0031-9155/40/11/013
    
    Kienle and Patterson, 1997, JOSA A. doi:10.1364/JOSAA.14.000246

    """

    if np.size(T)==1:
        steps = np.int32(np.floor((1.0*T)/dt))
        tvec = np.arange(steps)*dt + dt*0.5
    else:
        tvec = T
        dt = tvec[1]-tvec[0]
        steps = np.size(T)
    
    z0 = 1./(mua + musp) # in mm
    c0 = 299792458000. # mm/s; light spead in vacuum
    cm = c0/n
    D = 1./(3. * (mua + musp)) # mm; diffusion coefficient
    
    phi = np.zeros((steps, 2))
    phi[:,0] = tvec
    if z==0:
        # reflectance
        if boundary.lower() == 'zbc':
            phi[:,1] = (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * z0 * np.exp(-mua*cm*tvec) * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec))
        elif boundary.lower() == 'ebc-exact':
            A = utils.boundary_attenuation(n, method='exact')
            zp = z0 + 4*A*D
            phi[:,1] = 0.5 * (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * np.exp(-mua*cm*tvec) * (z0 * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec)) + zp * np.exp(-(zp**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'ebc-approx':
            A = utils.boundary_attenuation(n, method='approx')
            zp = z0 + 4*A*D
            phi[:,1] = 0.5 * (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * np.exp(-mua*cm*tvec) * (z0 * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec)) + zp * np.exp(-(zp**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'ebc-robin':
            A = utils.boundary_attenuation(n, method='robin')
            zp = z0 + 4*A*D
            phi[:,1] = 0.5 * (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * np.exp(-mua*cm*tvec) * (z0 * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec)) + zp * np.exp(-(zp**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'pcb-exact':
            A = utils.boundary_attenuation(n, method='exact')
            a = z0 * A/ (cm*tvec)
            b = 4/3 * A
            Tpcb = (1.0/a) * (1.0 - np.sqrt(np.pi/(a*b)) * np.exp((1.0+a)**2 / (a*b)) * special.erfc((1.0+a) / np.sqrt(a*b)))
            phi[:,1] = (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * z0 * np.exp(-mua*cm*tvec) * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec)) * Tpcb
        elif boundary.lower() == 'pcb-approx':
            A = utils.boundary_attenuation(n, method='approx')
            a = z0 * A/ (cm*tvec)
            b = 4/3 * A
            Tpcb = (1.0/a) * (1.0 - np.sqrt(np.pi/(a*b)) * np.exp((1.0+a)**2 / (a*b)) * special.erfc((1.0+a) / np.sqrt(a*b)))
            phi[:,1] = (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * z0 * np.exp(-mua*cm*tvec) * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec)) * Tpcb
        elif boundary.lower() == 'pcb-robin':
            A = utils.boundary_attenuation(n, method='robin')
            a = z0 * A/ (cm*tvec)
            b = 4/3 * A
            Tpcb = (1.0/a) * (1.0 - np.sqrt(np.pi/(a*b)) * np.exp((1.0+a)**2 / (a*b)) * special.erfc((1.0+a) / np.sqrt(a*b)))
            phi[:,1] = (4*np.pi*D*cm)**(-3/2) * tvec**(-5/2) * z0 * np.exp(-mua*cm*tvec) * np.exp(-(z0**2 + rho**2)/(4*D*cm*tvec)) * Tpcb
        else:
            raise ValueError('Unknown boundary type')
    else:
        # internal fluence
        if boundary.lower() == 'zbc':
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) - np.exp(-((z+z0)**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'ebc-exact':
            A = utils.boundary_attenuation(n, method='exact')
            zp = z0 + 4*A*D
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) - np.exp(-((z+zp)**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'ebc-approx':
            A = utils.boundary_attenuation(n, method='approx')
            zp = z0 + 4*A*D
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) - np.exp(-((z+zp)**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'ebc-robin':
            A = utils.boundary_attenuation(n, method='robin')
            zp = z0 + 4*A*D
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) - np.exp(-((z+zp)**2 + rho**2)/(4*D*cm*tvec)))
        elif boundary.lower() == 'pcb-exact':
            A = utils.boundary_attenuation(n, method='exact')
            zb = 2*A*D
            bnd = np.zeros(steps)
            for i in range(steps):
                bnd[i],_ = integrate.quad(lambda l,zb,zz0,r,dct4:np.exp(-l/zb)*np.exp(-((zz0+l)**2+r**2)/dct4), 0, np.Inf, args=(zb,z+z0,rho,4*D*cm*tvec[i]))
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) + np.exp(-((z+z0)**2 + rho**2)/(4*D*cm*tvec)) - 2/zb *bnd)
        elif boundary.lower() == 'pcb-approx':
            A = utils.boundary_attenuation(n, method='approx')
            zb = 2*A*D
            bnd = np.zeros(steps)
            for i in range(steps):
                bnd[i],_ = integrate.quad(lambda l,zb,zz0,r,dct4:np.exp(-l/zb)*np.exp(-((zz0+l)**2+r**2)/dct4), 0, np.Inf, args=(zb,z+z0,rho,4*D*cm*tvec[i]))
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) + np.exp(-((z+z0)**2 + rho**2)/(4*D*cm*tvec)) - 2/zb *bnd)
        elif boundary.lower() == 'pcb-robin':
            A = utils.boundary_attenuation(n, method='robin')
            zb = 2*A*D
            bnd = np.zeros(steps)
            for i in range(steps):
                bnd[i],_ = integrate.quad(lambda l,zb,zz0,r,dct4:np.exp(-l/zb)*np.exp(-((zz0+l)**2+r**2)/dct4), 0, np.Inf, args=(zb,z+z0,rho,4*D*cm*tvec[i]))
            phi[:,1] = cm * (4*np.pi*D*cm*tvec)**(-3/2) * np.exp(-mua*cm*tvec) * (np.exp(-((z-z0)**2 + rho**2)/(4*D*cm*tvec)) + np.exp(-((z+z0)**2 + rho**2)/(4*D*cm*tvec)) - 2/zb *bnd)
        else:
            raise ValueError('Unknown boundary type')
    phi[:,1] *= dt
    return phi
