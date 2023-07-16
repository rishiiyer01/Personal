

#∂²ψ/∂t² = c² * (∂²ψ/∂x² + ∂²ψ/∂y²)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup
nx, ny = 100, 100
nt = 200
c = 5
dx, dy = 0.5, 0.5
dt = 0.01

x = np.linspace(0, dx*(nx-1), nx)
y = np.linspace(0, dy*(ny-1), ny)

u = np.zeros((nx, ny))
unew = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

# Initial condition
u[int(0.5 / dx):int(1.0 / dx + 1), int(0.5 / dy):int(1.0 / dy + 1)] = 1

# Pre-compute all frames
frames = []
for it in range(nt):
    uold = u.copy()
    u_xminus = np.roll(u, 1, axis=0) # Shift +1 in x direction
    u_xplus = np.roll(u, -1, axis=0) # Shift -1 in x direction
    u_yminus = np.roll(u, 1, axis=1) # Shift +1 in y direction
    u_yplus = np.roll(u, -1, axis=1) # Shift -1 in y direction

    unew = 2 * u - uold + (c * dt / dx)**2 * (u_xplus - 2 * u + u_xminus) + \
            (c * dt / dy)**2 * (u_yplus - 2 * u + u_yminus)
    u = unew.copy()

    frames.append(u.copy())

# Animation
fig = plt.figure()
im = plt.imshow(frames[0], animated=True)

def updatefig(i):
    im.set_array(frames[i])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=50, blit=True)

plt.show()
