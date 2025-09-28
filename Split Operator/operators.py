import numpy as np
import math

# Here I define the parameters of the simulation, like the spatial grid, time step, total time, etc
class Parameters:
    def __init__(self, xmax:float, res: int, dt:float , timesteps: int, hbar: float, m: float, im_time: bool = False):
        self.xmax = xmax
        self.res = res
        self.dt = dt
        self.timesteps = timesteps
        self.im_time = im_time
        self.hbar = hbar
        self.m = m

        self.dx = 2 * xmax / res    
        self.dk = math.pi / xmax

        self.x = np.linspace(-xmax, xmax, res, endpoint=False)
        self.k = np.fft.fftfreq(res, d=self.dx) * 2*np.pi

# Here I define the operators that will perform the split-operator method
# in addition to the initial wave function

class Operators:
    def __init__(self, res: int):
        self.V = np.empty(res, dtype=complex)
        self.R = np.empty(res, dtype=complex)
        self.K = np.empty(res, dtype=complex)
        self.psi = np.empty(res, dtype=complex)