from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Mesh and function space
nx, ny = 100, 100
mesh = RectangleMesh(nx, ny, 5.0, 5.0)
V = FunctionSpace(mesh, "CG", 1)  # Continuous Galerkin space of linear elements

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Parameters
c = 15.0
dt = 0.001
nt = 500
f = 5.0
omega = 2.0 * np.pi * f

# Time-stepping variables
u_current = Function(V)
u_previous = Function(V)

# Weak form of the wave equation
a = u*v*dx + dt**2*c**2*dot(grad(u), grad(v))*dx
L = 2*u_current*v*dx - u_previous*v*dx

# Apply the driving function at the center of the domain
driving_force = Constant(0.0)
L += dt**2*driving_force*v*dx

# Boundary conditions (Neumann)
bcs = [DirichletBC(V, 0, (1, 2, 3, 4))]

# Solve
u_next = Function(V)
frames = []

for it in range(nt):
    # Driving function update
    if mesh.geometric_dimension() == 2:
        center = (2.5, 2.5)
    else:
        center = (2.5,)
    dist = sqrt(sum([(SpatialCoordinate(mesh)[i]-center[i])**2 for i in range(mesh.geometric_dimension())]))
    driving_force.assign(100*np.sin(omega * dt * it) * conditional(dist < 0.05, 1, 0))

    # Solve the equation
    solve(a == L, u_next, bcs=bcs)
    
    # Update time-stepping variables
    u_previous.assign(u_current)
    u_current.assign(u_next)

    # Append to frames
    frames.append(np.copy(u_next.dat.data))

# Animation
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(frames[0], animated=True, extent=[0, 5, 0, 5])
plt.colorbar(im, ax=ax)

def updatefig(i):
    im.set_array(frames[i])
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=len(frames), interval=50, blit=True)
HTML(ani.to_jshtml())