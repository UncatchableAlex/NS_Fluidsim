import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import cupy as np
from cupyx.scipy.sparse import diags,identity, kron
from cupyx.scipy.sparse.linalg import splu
from cupyx.scipy.interpolate import RegularGridInterpolator
from cupyx.scipy.interpolate import Akima1DInterpolator as interp1d



#import numpy as np
# from scipy.sparse import diags_array,identity, kron
# from scipy.sparse.linalg import splu
# from scipy.interpolate import RegularGridInterpolator, interp1d




#import numpy as np
# from scipy.sparse import diags_array,identity, kron
# from scipy.sparse.linalg import splu
# from scipy.interpolate import RegularGridInterpolator, interp1d

# Sources:
# Stable Fluids, Stam 1999   https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf
# Fast Fluid Dynamics Simulation on the GPU, Harris 2004           https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_GPU_Gems.pdf
def navier_stokes(x_range, n, t_max, iters, samples, init_U_func=lambda x,y: (0, 0), F=lambda X,Y,t: (X,Y,t), objects=lambda X,Y,t:np.zeros_like(X), viscosity=-20):
    dx = (x_range[1] - x_range[0]) / n
    dt = t_max / iters
    sample_freq = iters//samples
    # this is going to be the matrix that we use to capture "snapshots" of U throughout our simulation
    V = np.zeros((samples, n, n, 2), dtype=float)
    # this is going to be the matrix that we use to capture "snapshots" of the pressure throughout our simulation 
    P = np.zeros((samples, n, n), dtype=float)
    # this is the matrix that we will use to track the velocity of each discrete chunk of our simulation 
    U = np.zeros((3, n, n, 2), dtype=float)
    #sample_freq = iters/
    U[0] = np.array([[np.array(init_U_func(x_range[0] + dx*i, x_range[0] + dx*j)) for i in range(n)] for j in range(n)])
    U[1] = U[0]
    # construct indices for U
    X,Y = np.arange(0, U.shape[1]), np.arange(0, U.shape[2])
    row_idx, col_idx = np.meshgrid(X, Y, indexing='ij')
    idx = np.dstack((row_idx,col_idx))

    # make a matrix representing the descretized laplace operator:
    # https://en.wikipedia.org/wiki/Discrete_Poisson_equation
    # we have to decrement n because we will only be solving for our simulation's INTERIOR points. The boundary points
    # will be adjusted later in accordance with a boundary condition
    n -= 2
    diagonals = [
        [4] * n,
        [-1] * (n-1),
        [-1] * (n-1)
    ]
    offsets = np.array([0,1,-1])
    # this is the block triangular matrix that will compose the poisson matrix
    # Note that we use sparse arrays here because they are WAY too big to store literally
    D_sparse = diags(diagonals, offsets=offsets, format='csc')
    offdiag1 = diags([[-1] * (n**2 - n)], offsets=[n], format='csc')
    offdiag2 = diags([[-1] * (n**2 - n)], offsets=[-n], format='csc')
    # construct our final poisson matrix using a kronecker product
    # https://en.wikipedia.org/wiki/Kronecker_product
    poisson_mat_sparse = kron(identity(n), D_sparse, format='csc') + offdiag1 + offdiag2
    poisson_mat_sparse /= -(dx*dx)
    # calculate the inverse of our poisson matrix this will make solving this system repeatedly very easy
    start_t = time.time()
    poisson_inv = splu(poisson_mat_sparse)
    print(f'found poisson inv in {time.time() - start_t}')
    # make a matrix representing the implicit diffusion step:
    sparse_diffusion_matrix = identity(n*n, format='csc') - viscosity*dt*poisson_mat_sparse
    # find its inverse (we will be solving this system repeatedly, so this makes sense)
    start_t = time.time()
    diffusion_inv = splu(sparse_diffusion_matrix)
    print(f'found poisson inv in {time.time() - start_t}')
    # increment n back to what it was.
    n+=2
    # locate our objects:
    obj_mask = np.array(objects(x_range[0] + row_idx*dx, x_range[0] + col_idx*dx, 0))
    # run our simulation
    for i in tqdm(range(0, iters), total=iters, desc="Navier-Stokes Solver"):
        W0 = U[1]
        ########################################    FORCING      ################################################
        force =  F(x_range[0] + row_idx*dx, x_range[0] + col_idx*dx, i*dt)
        start_t = time.time()
        W0[:, :, 0] += force[0]
        W0[:, :, 1] += force[1]
        print(f'forcing in {time.time() - start_t}')
        U[2] = W1 = W0
        ########################################    ADVECTION    ################################################
        # find the index of each particle's velocity one step ago:
        start_t = time.time()
        shift = idx - (W1 * dt / dx)
        interp = RegularGridInterpolator((X, Y), W1, bounds_error=False, fill_value=0.0)
        #coords = shift.reshape((shift.shape[0]*shift.shape[1],2))
        U[2] = W2 = interp(shift)#interp(coords).reshape((shift.shape[0], shift.shape[1], 2))
        print(f'advection in {time.time() - start_t}')
        #######################################     DIFFUSION     #################################################
        # implicit derivation yayy:
        start_t = time.time()
        W2_x = diffusion_inv.solve(W2[1:-1, 1:-1, 0].flatten('F')).reshape(W2[1:-1, 1:-1, 0].shape, order='F')
        W2_y = diffusion_inv.solve(W2[1:-1, 1:-1, 1].flatten('F')).reshape(W2[1:-1, 1:-1, 1].shape, order='F')
        W2[1:-1, 1:-1, 0] = W2_x
        W2[1:-1, 1:-1, 1] = W2_y
        U[2] = W3 =  W2
        print(f'diff in {time.time() - start_t}')
        ######################################      PROJECTION     ######################################################
        # calculate pressure using the poisson pressure equation
        start_t = time.time()
        div_W3 = ((W3[2:, 1:-1, 0] - W3[0:-2, 1:-1, 0]) / (2*dx)) + ((W3[1:-1, 2:, 1] - W3[1:-1, :-2, 1]) /(2*dx))
        # solve for our pressure field
        p = poisson_inv.solve(div_W3.flatten('F')).reshape(div_W3.shape, order='F')
        # find the gradient of our pressure field
        div_px = (p[2:, 1:-1] - p[0:-2, 1:-1])/(2*dx)
        div_py = (p[1:-1, 2:] - p[1:-1, 0:-2])/(2*dx)
        div_p = np.dstack((div_px, div_py))
        U[2, 2:-2, 2:-2] = W3[2:-2, 2:-2] - div_p
        print(f'proj in {time.time() - start_t}')
        ######################################     BOUNDARIES & OBJECTS   ###############################################
        start_t = time.time()
        # update our boundary conditions like Harris in GPU Gems:
        U[2, 0:2, 2:-2] = 0#U[2, 3, 2:-2] # top boundary
        U[2, -3:-1, 2:-2] = 0# U[2, -3, 2:-2] # bottom boundary
        #U[2, 2:-2, 0:2] = U[2, 2:-2, 3:4] # left boundary
        U[2, 2:-2, -2:] = 0#U[2, 2:-2, -2:]  # right boundary
        U[2, obj_mask,:] = 0
        # shift 
        U[0] = U[1]
        U[1] = U[2]
        print(f'boundaries in {time.time() - start_t}')
        # if its time to save a sample, do it
        if i % sample_freq == 0:
            V[int(i // sample_freq)] = U[1]
            P[int(i // sample_freq), 1:-1, 1:-1] = p
        #return 0,0
    return V, P



def left(X,Y,t,amplitude=0.1):
    u = np.zeros_like(X)
    v = np.zeros_like(Y)
   # u = amplitude
    middle = np.average(Y[0])
    mags = -amplitude * np.linspace(Y[0][2] - middle, Y[0][-2] - middle, len(Y[0]) - 4)**2
    mags -= np.min(mags)
    rep_mags = np.repeat(mags[:, np.newaxis], repeats=10, axis=1)
    v[2:-2, 0:10] = rep_mags
    
    return u,v



def NACA_airfoil(X, Y, t, m=0.02, p=0.4, r=0.12):
    x = Y[0]
    t = r
    yc = (m / (p**2)) * (2 * p * x - x**2) * (x <= p) + (m / ((1 - p)**2)) * ((1 - 2 * p) + 2 * p * x - x**2) * (x > p)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    dyc_dx = (2 * m / (p**2)) * (p - x) * (x <= p) + (2 * m / ((1 - p)**2)) * (p - x) * (x > p)
    theta = np.arctan(dyc_dx)
    # Upper and lower surface coordinates
    offset = (x[-1] - x[0])/2
    xu = x - yt * np.sin(theta)#+ offset
    yu = yc + yt * np.cos(theta) + offset
    xl = x + yt * np.sin(theta)# + offset
    yl = yc - yt * np.cos(theta) + offset

    # Create the bit mask
    airfoil_mask = np.zeros_like(X, dtype=bool)
    interp_yl = interp1d(xl, yl)#, bounds_error=False, fill_value=0.0)
    interp_yu = interp1d(xu, yu)#, bounds_error=False, fill_value=0.0)
    
    airfoil_mask[(interp_yl(Y - (offset/4)) <= X) & (interp_yu(Y - (offset/4)) >= X)] = True
    #plt.imshow(airfoil_mask, cmap='Greys')
    #plt.show()
    #plt.plot(x, yt)
    #plt.plot(xl, yl, color='blue')
    #plt.plot(xu,yu)
    #plt.ylim((0,1))
    #plt.xlim((0,1))
    #plt.show()
    
    return airfoil_mask



from cupyx.profiler import profile
x_range = (0,2)
iters = samples = 20
n = 200
viscosity = 0
viscosity = 1.48 * 1e-5 # air at 15c
#viscosity = 0.001139  #water at 15c
# viscosity = 0.000282  #water at 100c
t_max = 20
dt = t_max / iters
dx = (x_range[1] - x_range[0]) / n
with profile():
    V, P = navier_stokes(x_range=x_range, n=n, t_max=t_max, iters=iters, samples=samples, viscosity=viscosity,init_U_func=lambda x,y: (0.1, 0) ,F=left, objects=NACA_airfoil)
    V = np.flip(V, axis=1)