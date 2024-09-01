#simulating 1d transient burgers equation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the parameters
L = 1.0  # Length of the domain
Nx = 100  # Number of grid points
dx = L / (Nx - 1)  # Grid spacing



class Burgers1D:
    def __init__(self, L=1.0, Nx=200, dt=0.001, T=1.0, nu=0.0005):
        self.L = L  # Length of the domain
        self.Nx = Nx  # Number of grid points
        self.dx = L / (Nx - 1)  # Grid spacing
        self.dt = dt  # Time step
        self.T = T  # Total simulation time
        self.nu = nu  # Kinematic viscosity
        
        self.x = np.linspace(0, L, Nx)  # Spatial grid
        self.u = np.zeros(Nx)  # Initial solution array
        self.u_history = []  # To store solution in time
        
    def set_initial_condition(self, func):
        self.u = func(self.x)
        self.u_history.append(self.u.copy())
        
    def solve(self):
        Nt = int(self.T / self.dt)  # Number of time steps
        A=np.zeros((self.Nx,self.Nx))
        for a in range(self.Nx):
            if a==0:
                A[a,a]=1/self.dx
                A[a,-2]=-1/self.dx
            else:
                A[a,a-1]=-1/self.dx
                A[a,a]=1/self.dx
            
        for n in range(Nt):
            # Compute spatial derivatives
            #du_dx = np.gradient(self.u, self.dx) #central difference
            #for upwinding:
            du_dx=np.dot(A,self.u)
            d2u_dx2 = np.gradient(du_dx, self.dx)#, i dont really want diffusion rn, but i do want anti diffusion
            # This is for a traffic simulation so we want to asymmetrically cause braking
            du_dx = np.where(du_dx < 0, 0.75 * du_dx, du_dx)
            # Update solution using forward Euler method
            #need to add noise
            self.u = self.u - self.u * self.dt * du_dx +np.random.randn(self.Nx)*0.01 -self.dt*d2u_dx2*self.nu
            self.u=np.where(self.u<0,0,self.u)
            # Apply periodic boundary conditions
            self.u[0] = self.u[-1]
            
            # Store the solution
            self.u_history.append(self.u.copy())
        
    def plot_solution(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [])
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, 2)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('1D Burgers Equation Solution')
        ax.grid(True)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            line.set_data(self.x, self.u_history[i])
            return line,

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(self.u_history), 
                             interval=50, blit=True)
        plt.show()



# Example usage:
if __name__ == "__main__":
    burgers = Burgers1D()
    #burgers.set_initial_condition(lambda x: np.sin(2 * np.pi * x)+1)
    burgers.set_initial_condition(lambda x: np.ones(x.shape))
    #gaussian initial condition
    #burgers.set_initial_condition(lambda x: np.exp(-(x-0.5)**2/0.01)+1)
    burgers.solve()
    burgers.plot_solution()