import numpy as np
from tqdm import tqdm
from scipy.sparse import diags_array,identity, kron
from scipy.sparse.linalg import splu
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import subprocess
import os 
from multiprocessing.pool import Pool


FRAME_FILE = 'frames.txt'

# Sources:
# Stable Fluids, Stam 1999   https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf
# Fast Fluid Dynamics Simulation on the GPU, Harris 2004           https://cg.informatik.uni-freiburg.de/intern/seminar/gridFluids_GPU_Gems.pdf
def navier_stokes(x_range, n, t_max, iters, num_particles, chunk_size, name, init_U_func=lambda x,y: (0, 0), F=[lambda X,Y,t: (X,Y,t)], viscosity=-20):
    plt.style.use('dark_background')
    num_particles = int(num_particles)
    dx = (x_range[1] - x_range[0]) / n
    dt = t_max / iters
    h = 2*dt

    # this is the matrix that we will use to track the velocity of each discrete chunk of our simulation 
    U = np.zeros((3, n, n), dtype=(np.double,2))
    #sample_freq = iters/
    U[0] = np.array([[init_U_func(x_range[0] + dx*i, x_range[0] + dx*j) for i in range(n)] for j in range(n)])
    U[1] = U[0]
    U[2] = U[0]


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
    # Note that we use sparse arrays here because they are WAY too big to store densely
    D_sparse = diags_array(diagonals, offsets=offsets, format='csc')
    offdiag1 = diags_array([[-1] * (n**2 - n)], offsets=[n], format='csc')
    offdiag2 = diags_array([[-1] * (n**2 - n)], offsets=[-n], format='csc')
    # construct our final poisson matrix using a kronecker product
    # https://en.wikipedia.org/wiki/Kronecker_product
    poisson_mat_sparse = kron(identity(n), D_sparse, format='csc') + offdiag1 + offdiag2
    poisson_mat_sparse /= -(dx*dx)
    # calculate the inverse of our poisson matrix this will make solving this system repeatedly very easy
    poisson_inv = splu(poisson_mat_sparse)
    # make a matrix representing the implicit diffusion step:
    sparse_diffusion_matrix = identity(n*n, format='csc') - viscosity*dt*poisson_mat_sparse
    # find its inverse (we will be solving this system repeatedly, so this makes sense)
    diffusion_inv = splu(sparse_diffusion_matrix)
    # increment n back to what it was.
    n+=2

    # create an array to track the position of a bunch of particles in the simulation
    particle_positions = np.random.uniform(low=x_range[0], high=x_range[1], size=(num_particles, 2))
    
    x_axis = y_axis = np.linspace(start=x_range[0], stop=x_range[1], num=U[0].shape[1])
    # We will timestep our particles using RK4
    #https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
    # interp0 = RegularGridInterpolator((x_axis, y_axis), U[0], bounds_error=False, fill_value=0.0)
    # interp1 = RegularGridInterpolator((x_axis, y_axis), U[1], bounds_error=False, fill_value=0.0)
    # interp2 = RegularGridInterpolator((x_axis, y_axis), U[2], bounds_error=False, fill_value=0.0)

    async_frames = []
    pool = Pool(processes=2)

    # try to make the temporary directory that we will store partial results in:
    try:
        os.mkdir('ffmpeg_temp')
    except:
        pass

    # run our simulation
    for i in tqdm(range(0, iters), total=iters, desc="Navier-Stokes Solver"):
        W0 = U[1]
        ########################################    FORCING      ################################################
        for f in F:
            W0 = f(W0, t=i*dt, t_max=t_max)
        # dummy_arr = np.zeros(W0[:,:,0].shape)[:,:, np.newaxis]
        # plt.imshow(np.dstack((W0, dummy_arr)))
        # plt.axis('off')  # Turn off the axis labels
        # plt.show()
        # raise Exception("stop here")
        U[2] = W1 = W0
        ########################################    ADVECTION    ################################################
        # find the index of each particle's velocity one step ago:
        shift = idx - (W1 * dt / dx)
        interp = RegularGridInterpolator((X, Y), W1, bounds_error=False, fill_value=0.0)
        U[2] = W2 = interp(shift)
        #######################################     DIFFUSION     #################################################
        # implicit derivation yayy:
        W2_x = diffusion_inv.solve(W2[1:-1, 1:-1, 0].flatten('F')).reshape(W2[1:-1, 1:-1, 0].shape, order='F')
        W2_y = diffusion_inv.solve(W2[1:-1, 1:-1, 1].flatten('F')).reshape(W2[1:-1, 1:-1, 1].shape, order='F')
        W2[1:-1, 1:-1, 0] = W2_x
        W2[1:-1, 1:-1, 1] = W2_y
        U[2] = W3 =  W2
        ######################################      PROJECTION     ######################################################
        # calculate pressure using the poisson pressure equation
        div_W3 = ((W3[2:, 1:-1, 0] - W3[0:-2, 1:-1, 0]) / (2*dx)) + ((W3[1:-1, 2:, 1] - W3[1:-1, :-2, 1]) /(2*dx))
        # solve for our pressure field
        p = poisson_inv.solve(div_W3.flatten('F')).reshape(div_W3.shape, order='F')
        # find the gradient of our pressure field
        div_px = (p[2:, 1:-1] - p[0:-2, 1:-1])/(2*dx)
        div_py = (p[1:-1, 2:] - p[1:-1, 0:-2])/(2*dx)
        div_p = np.dstack((div_px, div_py))
        U[2, 2:-2, 2:-2] = W3[2:-2, 2:-2] - div_p
        U[0] = U[1]
        U[1] = U[2]


        # # Trace particles:
        # k1 = interp0(particle_positions)
        # k2 = interp1(particle_positions + h*k1/2)
        # k3 = interp1(particle_positions + h*k2/2)
        # k4 = interp2(particle_positions + h*k3)

        # # use rk4 to guess the next position of every particle
        # particle_positions_new = particle_positions + h*(k1 + 2*k2 + 2*k3 + k4)/6
        # interp0 = interp1
        # interp1 = interp2
        # interp2 = RegularGridInterpolator((x_axis, y_axis), U[2], bounds_error=False, fill_value=0)

        # move stuck particles
        # mask = np.sum(particle_positions_new - particle_positions, axis=1) < 1e-3
        # stuck_particle_count = mask.sum()
        # if stuck_particle_count:
        #     # this will move stopped particles randomly
        #     particle_positions_new[mask, 0] = np.random.uniform(low=x_range[0], high=x_range[1], size=stuck_particle_count)
        #     particle_positions_new[mask, 1] = np.random.uniform(low=x_range[0], high=x_range[0] + ((x_range[1] - x_range[0])/15), size=stuck_particle_count)
            
        # particle_positions = particle_positions_new
        # start a process drawing a new frame:
        #draw_frame(x_range, np.copy(U[0]), None, np.copy(p))
        #raise Exception("stop here")
        new_frame_future = pool.apply_async(draw_frame, args=(x_range, np.copy(U[0]), None, np.copy(p)))
        async_frames.append(new_frame_future)
        if i % chunk_size == 0:
            save_frames_bin(async_frames, name)
            async_frames = []
    pool.close()
    pool.join()
    render2(async_frames, name)
    pool.terminate()

def save_frames_bin(async_frames, name):
    with open(FRAME_FILE, 'wb') as f:
        for async_frame in async_frames:
            f.write(async_frame.get())
    return

def render2(async_frames, name):
     #fig, (mag_ax, particle_ax, pressure_ax) = plt.subplots(nrows=3, ncols=1, figsize=(4, 14), sharex=True, subplot_kw={'xticks': [], 'yticks': []}, layout='constrained')
    fig, (mag_ax, pressure_ax) = plt.subplots(nrows=2, ncols=1, figsize=(4, 9), sharex=True, subplot_kw={'xticks': [], 'yticks': []}, layout='constrained')    
    # Define the initial frame
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=mag_ax, shrink=0.8, orientation='horizontal', pad=0.02)
    fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=pressure_ax, shrink=0.8, orientation='horizontal', pad=0.02)
    canvas_width, canvas_height = fig.canvas.get_width_height()
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-s', '%dx%d' % (canvas_width, canvas_height),
        '-pix_fmt', 'argb',
        '-r', '60',
        '-i', 'pipe:0',
        '-c:v', 'libx264',
        '-b:v', '0',
        '-crf', '10',
        '-preset', 'veryslow',
        f'{name}.mp4'
    ]
    # Run FFmpeg with input data piped to stdin
    sub_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with open(FRAME_FILE, 'rb') as f:  # Note: 'rb' for binary mode
        while True:
            chunk = f.read(8192)  # Read 8KB at a time
            if not chunk:
                break
            try:
                sub_process.stdin.write(chunk)
            except Exception as e:
                print(f"Error writing chunk {chunk}")
                stderr_data = sub_process.stderr.read()
                if stderr_data:
                    print(f"FFmpeg stderr: {stderr_data.decode('utf-8')}")
                break    

    # shove all of the frames still in memory into ffmpeg
    for async_frame in async_frames:
            try:
                sub_process.stdin.write(async_frame.get())
            except Exception as e:
                print(f"Error writing from memory")
                stderr_data = sub_process.stderr.read()
                if stderr_data:
                    print(f"FFmpeg stderr: {stderr_data.decode('utf-8')}")
                break    

    sub_process.stdin.close()
    stdout_data, stderr_data = sub_process.communicate()
    if sub_process.returncode != 0:
        print(f"FFmpeg error: {stderr_data.decode('utf-8')}")
    # close all figures that are still open for any reason:
  #  plt.close('all')
    return

# Function to update the plot
def draw_frame(x_range, u, particle_positions, pressure):
    u = np.flip(u, axis=0)
    pressure = np.flip(pressure, axis=0)
    # define the figure that we will plot with
    #fig, (mag_ax, particle_ax, pressure_ax) = plt.subplots(nrows=3, ncols=1, figsize=(4, 14), sharex=True, subplot_kw={'xticks': [], 'yticks': []}, layout='constrained')
    fig, (mag_ax, pressure_ax) = plt.subplots(nrows=2, ncols=1, figsize=(4, 9), sharex=True, subplot_kw={'xticks': [], 'yticks': []}, layout='constrained')
    
    # Define the initial frame
    mag_ax.clear()
    mag_ax.set_title("Speed (m/s)")
    norm = np.sqrt(u[:, :, 0]**2 + u[:, :, 1]**2)  # Calculate the magnitude of the vectors
    # Plot the vector field magnitude
    mag_ax.imshow(norm, cmap='viridis', extent=[x_range[0], x_range[1], x_range[0], x_range[1]])
    mag_ax.set_xlim(x_range[0], x_range[1])
    mag_ax.set_ylim(x_range[0], x_range[1])
    mag_ax.set_xticks([])
    mag_ax.set_yticks([])
    
    # particle_ax.clear()
    # particle_ax.set_title("Particle Trace")
    # particle_ax.set_xlim(x_range[0], x_range[1])
    # particle_ax.set_ylim(x_range[0], x_range[1])
    # particle_ax.scatter(particle_positions[:, 1], particle_positions[:, 0], s=0.002, color='hotpink')
    # particle_ax.set_xticks([])
    # particle_ax.set_yticks([])

    pressure_ax.clear()
    pressure_ax.set_title("Pressure (Pa)")
    pressure_ax.imshow(pressure, cmap='viridis', extent=[x_range[0], x_range[1], x_range[0], x_range[1]])
    pressure_ax.set_xlim(x_range[0], x_range[1])
    pressure_ax.set_ylim(x_range[0], x_range[1])
    # remove annoying tick marks:
    pressure_ax.set_xticks([])
    pressure_ax.set_yticks([])
    
    # add colorbars:
    vmin_mag = 0#np.min(norm[:, 50:])
    vmax_mag = 5#np.max(norm[:, 50:])
    vmin_pressure = -0.02#np.min(pressure[100:-100, 50:])
    vmax_pressure = 0.04 #np.max(pressure[100:-100, 50:])
    fig.colorbar(cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=vmin_mag, vmax=vmax_mag)), ax=mag_ax, shrink=0.8, orientation='horizontal', pad=0.02)
    fig.colorbar(cm.ScalarMappable(cmap='viridis', norm=colors.Normalize(vmin=vmin_pressure, vmax=vmax_pressure)), ax=pressure_ax, shrink=0.8, orientation='horizontal', pad=0.02)
    fig.canvas.draw()
    res = fig.canvas.tostring_argb()
    plt.close(fig)
    return res