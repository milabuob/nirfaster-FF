"""
Functions for basic data visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

def plotimage(mesh, data=None, cmap='hot'):
    """
    Fast preview of data within a 2D FEM mesh in the nirfasterff format.
    
    Plots an image of the values on the mesh. For 3D mesh, use :func:`~nirfasterff.visualize.plot3dmesh()` instead

    Parameters
    ----------
    mesh : nirfasterff mesh type
        a 2D nirfasterff mesh to plot the data on.
    data : double NumPy vector, optional
        data to be plotted, with size (NNode,). If not specified, treated as all zero.
    cmap : colormap used to plot the mesh. Default is 'hot'

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
    h.set_cmap(cmap)
    ax.view_init(90,-90,0)
    fig = plt.gcf()
    ax.axis('off')
    ax.axis('equal')
    fig.colorbar(h, location='bottom')
    # plt.show()   
    return fig, ax

def plot3dmesh(mesh, data=None, selector=None, alpha=0.8):
    """
    OLD IMPLEMENTATION USING MATPLOTLIB, CAN BE SLOW RENDERING LARGE MESHES.
    CONSIDER USING :func:`~nirfasterff.visulize.plot3dmesh_v2()` INSTEAD
    
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
        opacity, between 0-1. Default is 0.8

    Raises
    ------
    TypeError
        if mesh not 3D.

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

def plot3dmesh_v2(mesh, data=None, selector=None, alpha=1, cmap = 'Hot', clim=None):
    '''
    Fast preview of data within a 3D FEM mesh in the nirfasterff format, using Plotly as engine.
    
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
    
        opacity, between 0-1. The default is 1.
        
    cmap : str, optional
    
        colormap to use. See The default is 'Hot'.
        
    clim : array-like, optional
    
        colorlimit of the plot in format [cmin, cmax]. 
        
        The default is None, where min and max of data is used.

    Raises
    ------
    TypeError
        if mesh not 3D.

    Returns
    -------
    None

    '''
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
    # create mesh plot
    colors = np.max(data[np.int32(bndfaces-1)], axis=1)
    if np.all(clim==None):
        meshplot = go.Mesh3d(x=nodes[:,0], y=nodes[:,1], z=nodes[:,2],
                             i=bndfaces[:,0]-1, j=bndfaces[:,1]-1, k=bndfaces[:,2]-1, 
                             intensity=colors, colorscale=cmap, intensitymode='cell',
                             #facecolor=cmap((colors - colors.min())/(colors.max() - colors.min())), 
                             flatshading=True, showscale=True, opacity=alpha,
                             lighting=dict(specular=0),
                             colorbar=dict(orientation='h',xanchor='center',yanchor='bottom',y=0, len=0.4))
    else:
        meshplot = go.Mesh3d(x=nodes[:,0], y=nodes[:,1], z=nodes[:,2],
                             i=bndfaces[:,0]-1, j=bndfaces[:,1]-1, k=bndfaces[:,2]-1, 
                             intensity=colors, colorscale=cmap, intensitymode='cell',
                             #facecolor=cmap((colors - colors.min())/(colors.max() - colors.min())), 
                             flatshading=True, showscale=True, opacity=alpha, cmin=clim[0], cmax=clim[1],
                             lighting=dict(specular=0),
                             colorbar=dict(orientation='h',xanchor='center',yanchor='bottom',y=0,len=0.4))
    # create the frame wires
    facenodes = nodes[bndfaces.astype(int)-1]
    # borrowed from https://community.plotly.com/t/show-edges-of-the-mesh-in-a-mesh3d-plot/33614/3
    Xe = []
    Ye = []
    Ze = []
    for T in facenodes:
        Xe.extend([T[k%3][0] for k in range(4)]+[ None])
        Ye.extend([T[k%3][1] for k in range(4)]+[ None])
        Ze.extend([T[k%3][2] for k in range(4)]+[ None])
    wires = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines',
                         line=dict(color= [0.5,0.5,0.5], width=0.3))
    fig = go.Figure(data = [meshplot, wires])
    fig.update_layout(scene = dict(
        xaxis = dict(showticklabels = False, visible = False),
        yaxis = dict(showticklabels = False, visible = False),
        zaxis = dict(showticklabels = False, visible = False)))
    fig.show()
    
def plotvol(mesh, data, bnd=False, cmap='hot', clim=None, surfcnt=25, surfalpha=0.1):
    '''
    Renders volumetric data which is represented in the voxel space defined in mesh.vol.
    Can be used to render e.g. voxels-space fluence or Jacobian bananas
    
    Plotly is used as the engine, which essentially renders volumetric data using isosurfaces
    
    When 'bnd' is set to True, the outmost surface of the mesh is also plotted as a wire frame.
    This can take a few seconds when plotting large meshes.

    Parameters
    ----------
    mesh : nirfasterff mesh
    
        mesh that defines the voxel space as well as the outer boundary.
        
    data : NumPy array
    
        Can either be a 3D volume consistent with the volumetric space, or its vetorized representation (F order).
   
    bnd : bool, optional
    
        If true, also plots the outer boundadry of the mesh as a wire frame. The default is False.
    
    cmap : str, optional
    
        colormap to use. The default is 'hot'.
        
    clim : array-like, optional
    
        colorlimit of the plot in format [cmin, cmax]. The default is None.
        
    surfcnt : int, optional
    
        number of isosurfaces used in volume rendering.
        
        In general, more surfaces gives smoother results but can overshadow the details.
        The default is 25.
        
    surfalpha : float, optional
        opacity of the isosurfaces in volume rendering (0-1).
        
        A small value is desirable in order to see the inside. The default is 0.1.

    Raises
    ------
    TypeError
        if mesh is not 3D, or if mesh.vol is emtpy.
    ValueError
        if size of data is not the same as number of voxels

    Returns
    -------
    None.
    
    See Also
    --------
    Details about the parameters related to volume rendering:
        
        <https://plotly.com/python/3d-volume-plots/>
        
        <https://plotly.com/python-api-reference/generated/plotly.graph_objects.Volume.html>

    '''
    if mesh.dimension!=3:
        raise TypeError('Error: only 3D meshes are supported.')
    if not mesh.isvol():
        raise TypeError('Please run mesh.gen_intmat() or mesh.voxelize() first.')
        
    nvoxels = mesh.vol.xgrid.size * mesh.vol.ygrid.size * mesh.vol.zgrid.size
    if not nvoxels==data.size:
        raise ValueError('Size of data should be consistent with the voxel space')
    # if volumetric data, flatten it
    if np.ndim(data)==3:
        data = data.flatten('F')
    # volume rendering using plotly
    X,Y,Z = np.meshgrid(mesh.vol.xgrid, mesh.vol.ygrid, mesh.vol.zgrid)
    if np.all(clim==None):
        volplot = go.Volume(
            x=X.flatten('F'), y=Y.flatten('F'), z=Z.flatten('F'),
            value=data,
            opacity=surfalpha, # needs to be small to see through all surfaces
            surface_count=surfcnt, # needs to be a large number for good volume rendering
            colorscale=cmap, flatshading=True,
            colorbar=dict(orientation='h',xanchor='center',yanchor='bottom',y=0,len=0.4))
    else:
        volplot = go.Volume(
            x=X.flatten('F'), y=Y.flatten('F'), z=Z.flatten('F'),
            value=data,
            opacity=surfalpha, # needs to be small to see through all surfaces
            surface_count=surfcnt, # needs to be a large number for good volume rendering
            colorscale=cmap, flatshading=True,
            isomin=clim[0], isomax=clim[1],
            colorbar=dict(orientation='h',xanchor='center',yanchor='bottom',y=0,len=0.4))
    if not bnd:
        fig = go.Figure(data=volplot)
    else:
        ele = mesh.elements
        nodes = mesh.nodes
        faces = np.r_[ele[:, [0,1,2]], 
                      ele[:, [0,1,3]],
                      ele[:, [0,2,3]],
                      ele[:, [1,2,3]]]
        faces = np.sort(faces)
        # boundary faces: they are referred to only once
        faces,cnt=np.unique(faces,return_counts=1,axis=0)
        bndfaces=faces[cnt==1,:]
        # create the frame wires
        facenodes = nodes[bndfaces.astype(int)-1]
        # borrowed from https://community.plotly.com/t/show-edges-of-the-mesh-in-a-mesh3d-plot/33614/3
        Xe = []
        Ye = []
        Ze = []
        for T in facenodes:
            Xe.extend([T[k%3][0] for k in range(4)]+[ None])
            Ye.extend([T[k%3][1] for k in range(4)]+[ None])
            Ze.extend([T[k%3][2] for k in range(4)]+[ None])
        wires = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines',
                             line=dict(color= [0.5,0.5,0.5], width=0.3))
        fig = go.Figure(data=[volplot, wires])
    # turn off axes
    fig.update_layout(scene = dict(
        xaxis = dict(showticklabels = False, visible = False),
        yaxis = dict(showticklabels = False, visible = False),
        zaxis = dict(showticklabels = False, visible = False)))
    fig.show()
    

