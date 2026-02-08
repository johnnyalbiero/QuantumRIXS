import numpy as np
import math

""" Utility functions for Split Operator Method """

def next_fast_len(target: int) -> int:
    if target <= 0:
        return 1
    while True:
        n = target
        for p in [2, 3, 5]:
            while n % p == 0:
                n //= p
        if n == 1:
            return target
        target += 1

class Parameters:
    def __init__(self, xmin: float, xmax: float, dx: float, 
                 dt: float , timesteps: int, steps_per_frame: int, 
                 hbar: float, m: float, omega: float, res: int = 0, 
                 im_time: bool = False, renormalize: bool = False
    ):
        self.xmax = xmax
        self.xmin = xmin
        self.dt = dt
        self.timesteps = timesteps
        self.steps_per_frame = steps_per_frame
        self.im_time = im_time
        self.hbar = hbar
        self.m = m
        self.omega = omega
        self.renormalize = renormalize

        width = float(xmax) - float(xmin)

        if res <= 0:
            if dx > 0:
                res_candidate = math.ceil(width / float(dx))
            else:
                print("Warning: Invalid dx value. Using default resolution of 2048 points.")
                res_candidate = 2048 
        else:
            res_candidate = int(res)

        self.res = int(next_fast_len(res_candidate)) 
        
        self.dx = width / float(self.res)
        self.dk = 2 * math.pi / width 
        self.x = np.linspace(xmin, xmax, self.res, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(self.res, d=self.dx)

class Operators:
    def __init__(self, res: int):
        self.V = np.empty(res, dtype=complex)
        self.R = np.empty(res, dtype=complex)
        self.K = np.empty(res, dtype=complex)
        self.psi = np.empty(res, dtype=complex)
        self.corr = np.empty(res, dtype=complex)


""" Now here the physics start..."""

def init(par: Parameters, V_func, psi_func):
    opr = Operators(len(par.x))
    opr.V = V_func(par.x, par).astype(complex)
    opr.psi = psi_func(par.x, par).astype(complex)

    norm=np.sqrt(np.sum(np.abs(opr.psi) ** 2) * par.dx)
    if norm!= 0:
        opr.psi /= norm

    if par.im_time:
        opr.K = np.exp(-0.5 * par.dt * (par.hbar * par.k ** 2)/par.m)
        opr.R = np.exp(-0.5 * par.dt * opr.V / par.hbar)
    else:
        opr.K = np.exp(-0.5j * par.dt * (par.hbar * par.k ** 2)/par.m)
        opr.R = np.exp(-0.5j * par.dt * opr.V / par.hbar)
    return opr

def step(par: Parameters, opr: Operators):
    opr.psi *= opr.R
    opr.psi = np.fft.fft(opr.psi)
    opr.psi *= opr.K
    opr.psi = np.fft.ifft(opr.psi)
    opr.psi *= opr.R

    if par.im_time:
        density = np.abs(opr.psi) ** 2
        renorm_factor = sum(density) * par.dx
        opr.psi /= np.sqrt(renorm_factor)
    return np.abs(opr.psi) ** 2
    
def time_correlation(par: Parameters, opr: Operators, psi0: np.ndarray):
    opr.corr = np.conj(psi0) * opr.psi
    C_t = np.sum(opr.corr) * par.dx
    return C_t


def frequency_correlation_from_ct(par: Parameters, c_full: np.ndarray):
    freq = np.fft.fftshift(np.fft.fftfreq(len(c_full), d=par.dt)) * 2 * np.pi
    Cw = np.fft.fftshift(np.fft.ifft(c_full , norm = "forward")) * par.dt 
    return freq, Cw

def calculate_energy(par: Parameters, opr: Operators):
    psi_r = opr.psi
    psi_k = np.fft.fft(psi_r)
    psi_c = np.conj(psi_r)

    kinetic_operator_k_space = (par.hbar**2 * par.k**2) / (2 * par.m)
    energy_k = psi_c * np.fft.ifft(kinetic_operator_k_space * psi_k)
    energy_r = psi_c * opr.V * psi_r

    total_energy = np.sum(energy_k + energy_r) * par.dx
    return total_energy.real

def run_save_simulation(par: Parameters, opr: Operators, 
                        filename: str, n_frames: int, steps_per_frame: int):

    total_steps = max(1, min(par.timesteps, n_frames * max(1, steps_per_frame)))
    target_resolution = 2**13
    full_resolution = len(opr.psi)
    
    if full_resolution > target_resolution:
        stride = full_resolution // target_resolution
    else:
        stride = 1
        
    print(f"Original Resolution: {full_resolution} points.")
    print(f"Saving on each {stride} points (Video resolution: {full_resolution // stride} points).")
    
    psi_snapshots = np.zeros((n_frames + 1, len(opr.psi[::stride])), dtype=complex)
    correlation_history = np.zeros(total_steps + 1, dtype=complex)
    psi0 = opr.psi.copy()
    
    psi_snapshots[0] = psi0[::stride]
    correlation_history[0] = time_correlation(par, opr, psi0)
    
    energy = calculate_energy(par, opr).real
    
    print(f"initiating simulation: {total_steps} steps...")
    
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

    np.savez_compressed(filename, psi_snapshots=psi_snapshots, 
                        correlation_history=correlation_history, 
                        V=V_saved, x=x_saved, dt=par.dt, energy=energy, 
                        xmin=par.xmin, xmax=par.xmax)
    print("Finished.")
    