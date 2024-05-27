from ns import navier_stokes, left, NACA_airfoil

def test():
    x_range = (0,1.2)
    iters = 100
    n = 100
    viscosity = 0
    viscosity = 1.48 * 1e-5 # air at 15c
    #viscosity = 0.001139  #water at 15c
    # viscosity = 0.000282  #water at 100c
    t_max = 5
    navier_stokes(x_range=x_range, n=n, t_max=t_max, iters=iters, num_particles=1e5, chunk_size=10, name='wing6', viscosity=viscosity,init_U_func=lambda x,y: (0.3, 0) ,F=left, objects=NACA_airfoil)


if __name__ == '__main__':
    test()





