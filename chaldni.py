import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup
nx, ny = 100, 100
nt = 500
c = 5
dx, dy = 0.01, 0.01
dt = 0.001

x = np.linspace(0, dx*(nx-1), nx)
y = np.linspace(0, dy*(ny-1), ny)
X, Y = np.meshgrid(x, y)
#force=np.sin(5*np.pi*X)*np.sin(5*np.pi*Y)
u = np.zeros((nx, ny))
u=np.sin(10*np.pi*X)*np.sin(10*np.pi*Y)
unew = np.zeros((nx, ny))
uold = np.zeros((nx, ny))

# Driving frequency
f = 5.0
omega = 2.0 * np.pi * f

# Pre-compute all frames
frames = []
for it in range(nt):
    # Compute Neumann boundary conditions
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]

    # Apply the driving function at the center of the domain
    #u[nx//2, ny//2] += np.sin(omega * dt * it) * dt**2

    u_xminus = np.roll(u, 1, axis=0)
    u_xplus = np.roll(u, -1, axis=0)
    u_yminus = np.roll(u, 1, axis=1)
    u_yplus = np.roll(u, -1, axis=1)

    unew = 2 * u - uold + (c * dt / dx)**2 * (u_xplus - 2 * u + u_xminus) + \
           (c * dt / dy)**2 * (u_yplus - 2 * u + u_yminus)

    uold, u = u.copy(), unew.copy()

    frames.append(u.copy())

# Animation
fig = plt.figure()
im = plt.imshow(frames[0], animated=True, extent=[0, dx*nx, 0, dy*ny], origin='lower',cmap='magma')

def updatefig(i):
    im.set_array(frames[i])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=50, blit=True)
plt.show()