import numpy as np
from split import init, Operators, Parameters
from animation import animate_simulation

# Defining the potential and initial wave function for the simulation
def potencial(x, voffset: float = 0):
    #harmonic potential 1/2 (x - x0)^2
    #return 0.5 * (x - voffset) ** 2 

    # Square potential well
    #return np.piecewise(x, [x < -1, (x >= -1) & (x <= 1), x > 1], [0, 5, 0])

    # Morse potential
    D_e = 2.7  # Depth of the well
    a =  0.09  # Width parameter
    return D_e * (1 - np.exp(-a * (x - voffset)))**2 

def wavefunc(x, psioffset: float = -3.3):
    return (1 / (2 * np.pi) ** 0.25) * np.exp(-0.5 *(x - psioffset) ** 2, dtype=complex) # Psi(x) (1/(2pi)^(1/4)) * exp(- (x-x0)^2 / 2 )
    #return (1 / (2 * np.pi) ** 0.25) * np.exp(-0.5 * (x - psioffset) ** 2 + 1j * 5 * x, dtype=complex) # Psi(x) with initial momentum 

def main():
    # Simulation parameters
    par = Parameters(xmax=5, res=516, dt=0.01, timesteps=1500, hbar=1, m=1)
    opr = init(par, V_func=lambda x: potencial(x), psi_func=lambda x: wavefunc(x))

    # animation parameters 
    n_frames = 1500
    steps_per_frame = 1

    # Do the animation and saves in a mp4 or gif file
    animate_simulation(par, opr, n_frames, steps_per_frame, dpi=200)

if __name__ == "__main__":
    main()