

#∂²ψ/∂t² = c² * (∂²ψ/∂x² + ∂²ψ/∂y²)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup
nx, ny = 100, 100
nt = 200
c = 1.0
dx, dy = 1.0, 1.0
dt = 0.01

x = np.linspace(0, dx*(nx-1), nx)
y = np.linspace(0, dy*(ny-1), ny)

u = np.zeros((nx, ny))
unew = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

# Initial condition
u[int(0.5 / dx):int(1.0 / dx + 1), int(0.5 / dy):int(1.0 / dy + 1)] = 1

fig = plt.figure()
im = plt.imshow(u, animated=True)

def updatefig(*args):
    global u, uold, unew
    for it in range(20):  # Update 20 timesteps per frame
        uold = u.copy()
        for ix in range(1, nx-1):
            for iy in range(1, ny-1):
                unew[ix, iy] = 2 * u[ix, iy] - uold[ix, iy] + \
                    (c * dt / dx)**2 * (u[ix+1, iy] - 2 * u[ix, iy] + u[ix-1, iy]) + \
                    (c * dt / dy)**2 * (u[ix, iy+1] - 2 * u[ix, iy] + u[ix, iy-1])
        u = unew.copy()

    im.set_array(u)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)

plt.show()