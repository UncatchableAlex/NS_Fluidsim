from ns import navier_stokes
import numpy as np
from scipy.interpolate import interp1d
import matplotlib as plt
from matplotlib import cm, colors


def boundaries(u, row_idx, col_idx, t):
   # u[2:-2, -2:] = 0
    return u

def left(u, row_idx, col_idx, t, amplitude=5):
    u[:, 0:5, 0] = amplitude
    return u


def _NACA_airfoil(x, t, m, p ,r):
    t = r
    yc = (m / (p**2)) * (2 * p * x - x**2) * (x <= p) + (m / ((1 - p) ** 2)) * (
        (1 - 2 * p) + 2 * p * x - x**2
    ) * (x > p)
    yt = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    dyc_dx = (2 * m / (p**2)) * (p - x) * (x <= p) + (2 * m / ((1 - p) ** 2)) * (
        p - x
    ) * (x > p)
    theta = np.arctan(dyc_dx)
    # Upper and lower surface coordinates
    offset = (x[-1] - x[0]) / 2
    xu = x - yt * np.sin(theta)  # + offset
    yu = yc + yt * np.cos(theta) + offset
    xl = x + yt * np.sin(theta)  # + offset
    yl = yc - yt * np.cos(theta) + offset

    # Create the bit mask
    interp_yl = interp1d(xl, yl, bounds_error=False, fill_value=0.0)
    interp_yu = interp1d(xu, yu, bounds_error=False, fill_value=0.0)
    return interp_yl, interp_yu



def NACA_airfoil(u, row_idx, col_idx, t, m=0.06, p=0.4, r=0.09):
    x = col_idx[0]
    interp_yl, interp_yu = _NACA_airfoil(x, t, m, p, r)
    offset = (x[-1] - x[0]) / 2
    u[(interp_yl(col_idx - (offset / 4)) <= row_idx) & (interp_yu(col_idx - (offset / 4)) >= row_idx)] = 0
    return u



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


def test():
    x_range = (0, 1.2)
    iters = 100
    n = 100
    viscosity = 0
    viscosity = 1.48 * 1e-5  # air at 15c
    # viscosity = 0.001139  #water at 15c
    # viscosity = 0.000282  #water at 100c
    t_max = 5
    navier_stokes(
        x_range=x_range,
        n=n,
        t_max=t_max,
        iters=iters,
        num_particles=1e5,
        chunk_size=10,
        name="wing7",
        viscosity=viscosity,
        init_U_func=lambda x, y: (0.3, 0),
        F=[left],
       # draw_frame=draw_frame
    )






if __name__ == "__main__":
    test()

