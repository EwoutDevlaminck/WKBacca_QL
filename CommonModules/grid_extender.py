"""
This module provdes a fucntion which extends a scalar field given on a regural 
2D grid, to a slightly larger grid. 
"""


import numpy as np



# Auxiliary function for a three-point extrapolation
def extend(u0, a, b, c, points, s=+1):

    """
    Quadratic extrapolation formula.
    """

    assert type(a)==type(b)==type(c)
    assert type(points)==np.ndarray
    assert len(points.shape)==1
    if type(a)==np.ndarray:
        assert a.shape==b.shape==c.shape
    else:
        pass
    
    du = points[1] - points[0]
    A = a
    B = s * 0.5 * (3*a - 4*b + c) / du
    C = (a + c - 2*b) / du**2

    if type(a)==float:
        extrap_values = A + B*(points-u0) + 0.5*C*(points-u0)**2
    elif type(a)==np.ndarray:
        nptu = points.size
        extrap_values = np.empty((nptu,) + A.shape)
        for iu in range(nptu):
            extrap_values[iu] = A + B*(points[iu]-u0) + 0.5*C*(points[iu]-u0)**2
    else:
        msg = 'a,b,c must be all floats or all ndarrays with the same shape.'
        raise RuntimeError(msg) 
    
    return extrap_values



# Main extender function
def extend_regular_grid_data(x, y, z, extend_by=(1,1,1,1)):

    """
    Extend data given on a regural grid in 2D by means of polynomial
    extrapolation.
    
    USAGE:
         x, y, z = extend_regular_grid_data(x, y, z, extend_by=[ne,nw,nn,ns])
    
    Input data:
    
         x = 1D array of size nx
         y = 1D array of size ny
         z = ND array of shape (nx, ny, ...)
         extend_by = list of the form [ne, nw, nn, ns]

    The arrays x,y are the coordinates of the nodes (x[i], y[j]) in the grid
    and z[i,j,...] is the corresponding value at the node (x[i],y[j]) on a 
    possibly multi-component quantity. The grid will by estended by ne points 
    on the left (east), nw points on the right (west), nn points on the top
    (north), and ns points on the bottom (south). 

    The result x, y, z are the extended coordinate array and grid data.

    The extension uses a quadratic polynomial, with a three-point stencil, 
    for the extrapolation in the east, west, north, and south sectors, 
    and the average of the extrapolation in the corners. E.g., the values
    in the north-west corner is the average of the extrapolated value from
    the north and west sector.
    """

    # Extracting some data from the input
    ne, nw, nn, ns = extend_by

    xmin = x.min()
    xmax = x.max()

    ymin = y.min()
    ymax = y.max()

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    nx = x.size
    ny = y.size

    assert nx >= 4 and ny >= 4
    assert z.shape[0] == nx and z.shape[1] == ny


    # Extend the coordinate vectors 
    xe = xmin - dx*ne
    xw = xmax + dx*nw

    ys = ymin - dy*ns
    yn = ymax + dy*nn

    add_points_east = np.linspace(xe, xmin-dx, ne)
    add_points_west = np.linspace(xmax+dx, xw, nw)
    add_points_north = np.linspace(ymax+dy, yn, nn)
    add_points_south = np.linspace(ys, ymin-dy, ns)

    _x = np.concatenate((add_points_east, x, add_points_west))
    _y = np.concatenate((add_points_south, y, add_points_north))

    _nx = _x.size
    _ny = _y.size

    # Define the extended array for the data
    s = z.shape[2:]
    _z = np.zeros((_nx, _ny)+s)*np.nan

    # Copy the original data
    _z[ne:ne+nx,ns:ns+ny] = z
    
    # Extend into the EAST sector
    if ne > 0:
        za = z[0,:]
        zb = z[1,:]
        zc = z[2,:]
        z_extended_east = extend(xmin, za, zb, zc, add_points_east, s=-1)
        _z[0:ne,ns:ns+ny] = z_extended_east
    else:
        pass
    
    # Extend into the WEST sector
    if nw > 0:
        za = z[-1,:]
        zb = z[-2,:]
        zc = z[-3,:]
        z_extended_west = extend(xmax, za, zb, zc, add_points_west, s=+1)
        _z[ne+nx:,ns:ns+ny] = z_extended_west
    else:
        pass
    
    # Extend into the NORTH sector
    if nn > 0:
        za = z[:,-1]
        zb = z[:,-2]
        zc = z[:,-3]
        z_extended_north = extend(ymax, za, zb, zc, add_points_north, s=+1)
        _z[ne:ne+nx,ns+ny:] = np.swapaxes(z_extended_north, 0, 1)
    else:
        pass
    
    # Extend into the SOUTH sector
    if ns > 0:
        za = z[:,0]
        zb = z[:,1]
        zc = z[:,2]
        z_extended_south = extend(ymin, za, zb, zc, add_points_south, s=-1)
        _z[ne:ne+nx,0:ns] = np.swapaxes(z_extended_south, 0, 1)
    else:
        pass
    
    # Extend into the NORTH-EAST corner
    # We extrapolate from both the north and the east sectors and
    # compute the average
    if nn > 0 and ne > 0:
        # ... extrapolation from the east sector ...
        za = _z[0:ne,ns+ny-1]
        zb = _z[0:ne,ns+ny-2]
        zc = _z[0:ne,ns+ny-3]
        z_NE_from_E = extend(ymax, za, zb, zc, add_points_north, s=+1)
        z_NE_from_E = np.swapaxes(z_NE_from_E, 0, 1)
        # ... extrapolation from the north sector ...
        za = _z[ne,ns+ny:]
        zb = _z[ne+1,ns+ny:]
        zc = _z[ne+2,ns+ny:]
        z_NE_from_N = extend(xmin, za, zb, zc, add_points_east, s=-1)
        # ... define the extrapolation as an average ...
        _z[0:ne,ns+ny:] = 0.5 * (z_NE_from_E + z_NE_from_N)
    else:
        pass
    
    # Extend into the NORTH-WEST corner
    # We extrapolate from both the north and the west sectors and
    # compute the average
    if nn > 0 and nw > 0:
        # ... extrapolation from the west sector ...
        za = _z[ne+nx:,ns+ny-1]
        zb = _z[ne+nx:,ns+ny-2]
        zc = _z[ne+nx:,ns+ny-3]
        z_NW_from_W = extend(ymax, za, zb, zc, add_points_north, s=+1)
        z_NW_from_W = np.swapaxes(z_NW_from_W, 0, 1)
        # ... extrapolation from the north sector ...
        za = _z[ne+nx-1,ns+ny:]
        zb = _z[ne+nx-2,ns+ny:]
        zc = _z[ne+nx-3,ns+ny:]
        z_NW_from_N = extend(xmax, za, zb, zc, add_points_west, s=+1)
        # ... define the extrapolation as an average ...
        _z[ne+nx:,ns+ny:] = 0.5 * (z_NW_from_W + z_NW_from_N)
    else:
        pass
    
    # Extend into the SOUTH-WEST corner
    # We extrapolate from both the south and the west sectors and
    # compute the average
    if ns > 0 and nw > 0:
        # ... extrapolation from the west sector ...
        za = _z[ne+nx:,ns]
        zb = _z[ne+nx:,ns+1]
        zc = _z[ne+nx:,ns+2]
        z_SW_from_W = extend(ymin, za, zb, zc, add_points_south, s=-1)
        z_SW_from_W = np.swapaxes(z_SW_from_W, 0, 1)
        # ... extrapolation from the south sector ...
        za = _z[ne+nx-1,0:ns]
        zb = _z[ne+nx-2,0:ns]
        zc = _z[ne+nx-3,0:ns]
        z_SW_from_S = extend(xmax, za, zb, zc, add_points_west, s=+1)
        # ... define the extrapolation as an average ...
        _z[ne+nx:,0:ns] = 0.5 * (z_SW_from_W + z_SW_from_S)
    else:
        pass
    
    # Extend into the SOUTH-EST corner
    # We extrapolate from both the north and the east sectors and
    # compute the average
    if ns > 0 and ne > 0:
        # ... extrapolation from the east sector ...
        za = _z[0:ne,ns]
        zb = _z[0:ne,ns+1]
        zc = _z[0:ne,ns+2]
        z_SE_from_E = extend(ymin, za, zb, zc, add_points_south, s=-1)
        z_SE_from_E = np.swapaxes(z_SE_from_E, 0, 1)
        # ... extrapolation from the south sector ...
        za = _z[ne,0:ns]
        zb = _z[ne+1,0:ns]
        zc = _z[ne+2,0:ns]
        z_SE_from_S = extend(xmin, za, zb, zc, add_points_east, s=-1)
        # ... define the extrapolation as an average ...
        _z[0:ne,0:ns] = 0.5 * (z_SE_from_E + z_SE_from_S)
    else:
        pass

        
    return _x, _y, _z 
    
    


# Testing
if __name__=='__main__':

    # Test cases:
    ## fun = lambda x, y: x**2
    ## fun = lambda x, y: y**2    
    ## fun = lambda x, y: 20*x**2 + y**2 - 3.*x
    fun = lambda x, y: 20*x**2 + y**2 - 3.*x + 10.*np.sin(3*y)

    nx, ny = 101, 100
    x = np.linspace(0., 1., nx)
    y = np.linspace(-2., +2., ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    z = fun(X, Y)

    added_pnts = ne, nw, nn, ns = (40,20,22,16)
    _x, _y, _z = extend_regular_grid_data(x, y, z, extend_by=added_pnts)

    print('\nShape of the original grid = {}'.format(z.shape))
    print('\nShape of the extended grid = {}'.format(_z.shape))
    print('\nAdded points = {}\n'.format(added_pnts))

    _X, _Y = np.meshgrid(_x, _y, indexing='ij')

    _z_exact = fun(_X, _Y)

    error = np.abs(_z - _z_exact)

    print('Min. error = {}\n'.format(error.min()))
    print('Max. error = {}\n'.format(error.max()))
    
    import matplotlib.pyplot as plt


    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    p0 = ax0.pcolormesh(x, y, z.T)
    ax0.set_title('original data')
    fig0.colorbar(p0)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    p1 = ax1.pcolormesh(_x, _y, _z.T)
    ax1.set_title('extrapolated')
    fig1.colorbar(p1)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    p2 = ax2.pcolormesh(_x, _y, error.T)
    ax2.set_title('error')
    fig2.colorbar(p2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    p3 = ax3.pcolormesh(_x, _y, _z_exact.T)
    ax3.set_title('exact')
    fig3.colorbar(p3)
    
    plt.show()



# End of file
