from ns import navier_stokes
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import rotate
import matplotlib as plt
from matplotlib import cm, colors


def boundaries(u, row_idx, col_idx, t):
   # u[2:-2, -2:] = 0
    return u

def left(u, t,t_max, amplitude=5):
    u[:, 0:30, 1] = amplitude
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
    offset_vert = (x[-1] - x[0]) / 2
    xu = x - yt * np.sin(theta)  # + offset
    yu = yc + yt * np.cos(theta)# + offset_vert
    xl = x + yt * np.sin(theta)  # + offset
    yl = yc - yt * np.cos(theta) #+ offset_vert

    # Create the bit mask
    interp_yl = interp1d(xl, yl, bounds_error=False, fill_value=0.0)
    interp_yu = interp1d(xu, yu, bounds_error=False, fill_value=0.0)
    return interp_yl, interp_yu



def NACA_airfoil(u, t, t_max, m=0.06, p=0.4, r=0.09, aoa=10):
    aoa = (-25*(t-2)/t_max)-3 if t > 2 else -3 
    y = np.linspace(-1,1,len(u))
    x = np.linspace(-1,1,len(u))

    x2,y2 = np.meshgrid(x,y)
    interp_yl, interp_yu = _NACA_airfoil(x, t, m, p, r)

    # transform our aoa to radians
    rad_aoa = np.deg2rad(aoa)

    # apply a rotation matrix (linear algebra moment)
    rot = lambda a, b: (a*np.cos(rad_aoa) - b*np.sin(rad_aoa), a*np.sin(rad_aoa) + b*np.cos(rad_aoa))
    rot_x_yl, rot_yl = rot(x,interp_yl(x+0.5)) # we translate the wing left by 0.5 to put its center at the origin (around which the rotation occurs)
    rot_x_yu, rot_yu = rot(x,interp_yu(x+0.5)) # this effectively results in the wing rotating around its center (as opposed to around the leading edge which was at the origin before)

    # make interpolators that capture our rotation transformation
    interp_rot_yl = interp1d(rot_x_yl, rot_yl, bounds_error=False, fill_value=0.0)
    interp_rot_yu = interp1d(rot_x_yu, rot_yu, bounds_error=False, fill_value=0.0)
    
    # apply our rotation interpolators to our x grid, and select coords in between the top and bottom of the wing
    u[(interp_rot_yl(x2) < y2) & (interp_rot_yu(x2) > y2)] = 0
    return u

def stats(u, p, row_idx, col_idx, t):
    pass



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
    x_range = (-1, 1)
    n = 300
    viscosity = 0
    viscosity = 1.48 * 1e-5  # air at 15c
    # viscosity = 0.001139  #water at 15c
    # viscosity = 0.000282  #water at 100c
    t_max = 15
    iters = 60 * t_max
    navier_stokes(
        x_range=x_range,
        n=n,
        t_max=t_max,
        iters=iters,
        num_particles=1e5,
        chunk_size=900,
        name="wing8",
        viscosity=viscosity,
        init_U_func=lambda x, y: (1, 0),
        F=[NACA_airfoil, left],
       # draw_frame=draw_frame
    )






if __name__ == "__main__":
    test()

