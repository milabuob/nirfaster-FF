"""
Functions for basic data visualization
"""
import numpy as np
import matplotlib.pyplot as plt

def plotimage(mesh, data=None):
    """
    Fast preview of data within a 2D FEM mesh in the nirfasterff format.
    
    Plots an image of the values on the mesh. For 3D mesh, use :func:`~nirfasterff.visualize.plot3dmesh()` instead

    Parameters
    ----------
    mesh : nirfasterff mesh type
        a 2D nirfasterff mesh to plot the data on.
    data : double NumPy vector, optional
        data to be plotted, with size (NNode,). If not specified, treated as all zero.

    Raises
    ------
    TypeError
        if mesh not 2D.

    Returns
    -------
    matplotlib.figure.Figure
        the figure to be displayed
    mpl_toolkits.mplot3d.axes3d.Axes3D
        Current axes of the plot. Can be subsequently used for further plotting.

    """
   
    if mesh.dimension!=2:
        raise TypeError('Error: only 2D meshes are supported.')
        
    if np.all(data==None):
        data = np.zeros(mesh.nodes.shape[0])
    ax = plt.figure().add_subplot(projection='3d')
    h = ax.plot_trisurf(mesh.nodes[:,0], mesh.nodes[:,1], np.zeros(mesh.nodes.shape[0]), triangles=mesh.elements-1, linewidth=None, edgecolor=None, antialiased=True)
    colors = np.max(data[np.int32(mesh.elements-1)], axis=1)
    h.set_array(colors)
    h.set_cmap('hot')
    ax.view_init(90,-90,0)
    fig = plt.gcf()
    ax.axis('off')
    ax.axis('equal')
    fig.colorbar(h, location='bottom')
    # plt.show()   
    return fig, ax

def plot3dmesh(mesh, data=None, selector=None, alpha=0.8):
    """
    Fast preview of data within a 3D FEM mesh in the nirfasterff format.
    
    Plots an image of the values on the mesh at the intersection specified by "selector". 
    
    For 2D mesh, use :func:`~nirfasterff.visualize.plotimage()` instead

    Parameters
    ----------
    mesh : nirfasterff mesh type
        a 3D nirfasterff mesh to plot the data on.
    data : double NumPy vector, optional
        data to be plotted, with size (NNode,). If not specified, treated as all zero.
    selector : str, optional
        Specifies the intersection at which the data will be plotted, e.g. 'x>50', or '(x>50) | (y<100)', or 'x + y + z < 200'.
        
        Note that "=" is not supported. When "|" or "&" are used, make sure that all conditions are put in parantheses separately
        
        If not specified, function plots the outermost shell of the mesh.
    alpha : float, optional
        transparency, between 0-1. Default is 0.8

    Raises
    ------
    TypeError
        if mesh not 2D.

    Returns
    -------
    matplotlib.figure.Figure
        the figure to be displayed
    mpl_toolkits.mplot3d.axes3d.Axes3D
        Current axes of the plot. Can be subsequently used for further plotting.
        
    Notes
    -------
    This function is adapted from the 'plotmesh' function in the iso2mesh toolbox
    
    https://iso2mesh.sourceforge.net/cgi-bin/index.cgi

    """

    if mesh.dimension!=3:
        raise TypeError('Error: only 3D meshes are supported.')
        
    if np.all(data==None):
        data = np.zeros(mesh.nodes.shape[0])
    ele = mesh.elements
    nodes = mesh.nodes
    x = np.mean(nodes[np.int32(ele-1),0], axis=1)
    y = np.mean(nodes[np.int32(ele-1),1], axis=1)
    z = np.mean(nodes[np.int32(ele-1),2], axis=1)
    
    if np.all(selector==None):
        idx = np.arange(mesh.elements.shape[0])
    else:
        # select the subset of elements
        idx = eval('np.nonzero(' + selector + ')[0]')
        
    faces = np.r_[ele[np.ix_(idx, [0,1,2])], 
                  ele[np.ix_(idx, [0,1,3])],
                  ele[np.ix_(idx, [0,2,3])],
                  ele[np.ix_(idx, [1,2,3])]]
    faces = np.sort(faces)
    # boundary faces: they are referred to only once
    faces,cnt=np.unique(faces,return_counts=1,axis=0)
    bndfaces=faces[cnt==1,:]
    
    # plot
    ax = plt.figure().add_subplot(projection='3d')
    h = ax.plot_trisurf(nodes[:,0], nodes[:,1], nodes[:,2], triangles=bndfaces-1, linewidth=0.2, edgecolor=[0.5,0.5,0.5], antialiased=True)
    h.set_alpha(alpha)
    colors = np.max(data[np.int32(bndfaces-1)], axis=1)
    h.set_array(colors)
    ax.axis('off')
    ax.axis('equal')
    fig = plt.gcf()
    fig.colorbar(h, location='bottom')
    # plt.show()
    return fig, ax
