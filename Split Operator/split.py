import numpy as np
import math

""" Class functions that sets all the relevants parameters and operators to the
Physics of the Split operator method.
"""

class Parameters:
    # Collection of all the necessary parameters for the simulation.
    def __init__(self, xmin: float, xmax: float, res: int, 
                 dt: float , timesteps: int, steps_per_frame: int, 
                 hbar: float, m: float, omega: float, Gamma: float,
                 im_time: bool = False
    ):
        
    # Receives all arguments set in main.py 
    # and constructs all other necessary parameters, such as the grid in the frequency domain.
        self.xmax = xmax
        self.xmin = xmin
        self.res = res
        self.dt = dt
        self.timesteps = timesteps
        self.steps_per_frame = steps_per_frame
        self.im_time = im_time
        self.hbar = hbar
        self.m = m
        self.omega = omega
        self.Gamma = Gamma

        width = float(xmax) - float(xmin)

        self.dx = width / self.res
        self.dk = 2 * math.pi / width 
        self.x = np.linspace(xmin, xmax, self.res, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(self.res, d=self.dx)

class Operators:
    def __init__(self, res: int):
        # Initializes arrays representing the operators and state vectors for the simulation.
        self.V = np.empty(res, dtype=complex)
        self.R = np.empty(res, dtype=complex)
        self.K = np.empty(res, dtype=complex)
        self.psi = np.empty(res, dtype=complex)
        self.corr = np.empty(res, dtype=complex)


""" Core functions for the time evolution simulation of the wavefunction.

This module initializes the necessary operators based on parameters from main.py 
and performs the propagation of the wavefunction, calculating the time correlation 
at each step.

Note: The Fourier transform of the time correlation is provided for post-processing 
in viewer.py and is not required for the simulation loop itself.
"""

def init(par: Parameters, V_func, psi_func):
    # For the split operator, this function receives the mathematical functions
    # from main.py and constructs the exponentials of kinetic and potential terms
    # in their respective domains.
    opr = Operators(len(par.x))
    opr.V = V_func(par.x, par).astype(complex)
    opr.psi = psi_func(par.x, par).astype(complex)

    norm = np.sqrt(np.sum(np.abs(opr.psi) ** 2) * par.dx)

    if norm != 0:
        opr.psi /= norm

    if par.im_time:
        opr.K = np.exp(-0.5 * par.dt * (par.hbar * par.k ** 2)/par.m)
        opr.R = np.exp(-0.5 * par.dt * opr.V / par.hbar)
    else:
        opr.K = np.exp(-0.5j * par.dt * (par.hbar * par.k ** 2)/par.m)
        opr.R = np.exp(-0.5j * par.dt * opr.V / par.hbar)
    return opr

def step(par: Parameters, opr: Operators):
    # Here the simulation begins. The wave function psi is applied to the kinetic and potential
    # terms using the FFT algorithm for the Fourier Transform to obtain the numerical values 
    # of each operator.
    opr.psi *= opr.R
    opr.psi = np.fft.fft(opr.psi)
    opr.psi *= opr.K
    opr.psi = np.fft.ifft(opr.psi)
    opr.psi *= opr.R
    
    # The following line applies the exponential of Gamma decay. Note that this exponential
    # is a constant term; it is included here optionally for the video simulation.
    opr.psi *= np.exp(- (par.Gamma / par.hbar) * par.dt)

    if par.im_time:
        # Renormalize Psi in each step if necessary.
        density = np.abs(opr.psi) ** 2
        renorm_factor = np.sum(density) * par.dx
        opr.psi /= np.sqrt(renorm_factor)
    return np.abs(opr.psi) ** 2
    
def time_correlation(par: Parameters, opr: Operators, psi0: np.ndarray):
    # Besides using the Split Operator method for the evolution of the wavefunction,
    # we calculate the time correlation <psi(0)|psi(t)> at each step.
    opr.corr = np.conj(psi0) * opr.psi 
    C_t = np.sum(opr.corr) * par.dx
    return C_t


def frequency_correlation_from_ct(par: Parameters, c_full: np.ndarray):
    # Uses the Time Correlation <psi(0)|psi(t)> to calculate the Fourier Transform
    # and obtain the spectrum of the Hamiltonian (Total Energy eigenvalues).
    freq = np.fft.fftfreq(len(c_full), d=par.dt) * 2 * np.pi
    d_omega = freq[1] - freq [0]
    Cw = (np.fft.ifft(np.fft.ifftshift(c_full), norm = "forward")) * (1 / (len(freq)* d_omega *par.hbar))
    return freq, Cw

def calculate_energy(par: Parameters, opr: Operators):
    # If necessary at some point, this function calculates the total energy
    # of the psi wavefunction.
    psi_r = opr.psi
    psi_k = np.fft.fft(psi_r)
    psi_c = np.conj(psi_r)

    kinetic_operator_k_space = (par.hbar**2 * par.k**2) / (2 * par.m)
    energy_k = psi_c * np.fft.ifft(kinetic_operator_k_space * psi_k)
    energy_r = psi_c * opr.V * psi_r

    total_energy = np.sum(energy_k + energy_r) * par.dx
    return total_energy.real

"""
Besides the simulation functions above, this function creates a raw archive (.npz) that
saves all relevant simulation values, such as psi values across the grid (for video plotting),
correlation values at each time step (for post-processing), and the Fourier Transform data
to obtain the spectrum.

It also saves the grid, the potential function applied to the grid, and the initial psi function.
This part runs the full simulation and compresses the results into a file that can be used by 
viewer.py to plot and post-process the data.
"""

def run_save_simulation(par: Parameters, opr: Operators, 
                        filename: str, n_frames: int, steps_per_frame: int):
    
    # Calculate the total steps performed in the simulation
    total_steps = max(1, min(par.timesteps, n_frames * max(1, steps_per_frame)))
    target_resolution = 2**16
    full_resolution = len(opr.psi)
    
    if full_resolution > target_resolution:
        stride = full_resolution // target_resolution
    else:
        stride = 1
        
    print(f"Original Resolution: {full_resolution} points.")
    print(f"Saving every {stride} points (Video resolution: {full_resolution // stride} points).")
    
    # Create and save the array of the most important values.
    psi_snapshots = np.zeros((n_frames + 1, len(opr.psi[::stride])), dtype=complex)
    correlation_history = np.zeros(total_steps + 1, dtype=complex)
    psi0 = opr.psi.copy()
    
    psi_snapshots[0] = psi0[::stride]
    correlation_history[0] = time_correlation(par, opr, psi0)
    
    energy = calculate_energy(par, opr).real
    
    print(f"Initiating simulation: {total_steps} steps...")
    
    frame_idx = 1
    for i in range(1, total_steps + 1):
        step(par, opr)
        correlation_history[i] = time_correlation(par, opr, psi0)
        if i % steps_per_frame == 0 and frame_idx <= n_frames:
            psi_snapshots[frame_idx] = opr.psi[::stride]
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Current frame: {frame_idx}/{n_frames}", end='\r')

    print(f"\nSaving data to {filename}...")
    
    x_saved = par.x[::stride]
    V_saved = opr.V[::stride]

    np.savez_compressed(filename, psi_snapshots = psi_snapshots, 
                        correlation_history = correlation_history, 
                        V = V_saved, x = x_saved, dt = par.dt, energy = energy, 
    )
    print("Finished.")