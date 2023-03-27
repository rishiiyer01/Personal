#spherical pendulum simulation
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameters
m = 1.0      # Mass of the pendulum
g = 9.81     # Acceleration due to gravity
L = 1.0      # Length of the pendulum rod

# Define the system dynamics using the Euler-Lagrange equations
def dynamics(y, t):
    theta, phi, theta_dot, phi_dot = y
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_theta_phi = np.sin(theta - phi)
    cos_theta_phi = np.cos(theta - phi)
    f_theta = -m * g * sin_theta - m * L * (theta_dot**2 * sin_theta - phi_dot**2 * sin_theta_phi * cos_theta_phi)
    f_phi = -m * g * sin_phi - m * L * (phi_dot**2 * sin_theta_phi + theta_dot**2 * cos_theta_phi) * sin_phi / L
    return [theta_dot, phi_dot, f_theta / (m * L**2), f_phi / (m * L**2)]

# Initial conditions
y0 = [np.pi/4, np.pi/4, 0.0, 0.0]     # theta, phi, theta_dot, phi_dot

# Time vector
t = np.linspace(0, 10, 1000)

# Integrate the system of equations
sol = odeint(dynamics, y0, t)

# Extract the solution
theta = sol[:, 0]
phi = sol[:, 1]

# Plot the motion of the pendulum
x = L * np.sin(theta) * np.cos(phi)
y = L * np.sin(theta) * np.sin(phi)
z = -L * np.cos(theta)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
