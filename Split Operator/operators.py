import numpy as np
import math

class Parameters:
    def __init__(self, xmax:float, res: int, dt:float , timesteps: int, im_time: bool = False):
        self.xmax = xmax
        self.res = res
        self.dt = dt
        self.timesteps = timesteps
        self.im_time = im_time

        self.dx = 2 * xmax / res    
        self.dk = math.pi / xmax

        self.x = np.linspace(-xmax, xmax, res, endpoint=False)
        self.k = np.fft.fftfreq(res, d=self.dx) * 2*np.pi

    
class Operators:
    ## Here I define the operators that will perform the split-operator method
    ## in addition to the initial wave function
    def __init__(self, res: int):
        self.V = np.empty(res, dtype=complex)
        self.R = np.empty(res, dtype=complex)
        self.K = np.empty(res, dtype=complex)
        self.psi = np.empty(res, dtype=complex)