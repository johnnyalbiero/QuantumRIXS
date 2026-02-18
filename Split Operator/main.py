import numpy as np
import os
from split import init, Parameters, run_save_simulation

""" 
This module defines the simulation parameters chosen by the user, including 
grid settings, resolution, time steps, and physical constants.

It also handles the selection of Potential functions and initial Wavefunctions.
Finally, it executes the simulation step-by-step and saves the results to a 
compressed .npz file.
"""

# Configuration of all necessary parameters
config = Parameters(
    xmin = -5, xmax = 5, 
    res = 2**17, # Since the algorithm uses FFT to calculate the dynamics, it is more efficient to use 2^n resolution.
    dt = 0.01, timesteps = 2**17, steps_per_frame = 100,
    hbar = 1, m = 1, omega = 5 , Gamma = 0.005
)

pot_params = {'voffset': 0.0}
psi_params = {'gamma': 0.1, 'k_0': -0.2, 'psioffset': 2}

# Potential energy functions and Wavefunction definitions.

def V_harmonic(x, par, voffset):
    return 0.5 * par.m * (par.omega**2) * (x - voffset)**2

def V_morse(x, par, D_e, a, voffset):
    return D_e * (1 - np.exp(-a * (x - voffset)))**2

    """ --- Note on bound states calculation ---
    lambda_ = np.sqrt(2 * par.m * D_e) / (a * par.hbar)
    n_max = int(lambda_ - 0.5)

    # for i in range(n_max + 1):
    #     E_n = (par.hbar * a)**2 / (2 * par.m) * (lambda_**2 - (lambda_ - i - 0.5)**2)
    #     print(f"Energy level n={i}: E_n = {E_n}")  
    """

def wavefunc(x, par, gamma, k_0, psioffset):
    # Gaussian wave packet
    # Psi(x) = 1/(2πγ²)^(1/4) * exp(-0.5 * ((x - x0)/γ)²)
    # return (1 / (2 * np.pi * gamma**2) ** 0.25) * np.exp(-0.25 * ((x - psioffset) / gamma) ** 2) 
    
    # Ground state of the Quantum Harmonic Oscillator
    # Psi(x) = (mω/πħ)^(1/4) * exp(-0.5 * mω/ħ * (x - x0)²)
    return (par.omega * par.m / (np.pi * par.hbar)) ** 0.25 * np.exp(-0.5 * (par.m * par.omega / par.hbar) * (x - psioffset) ** 2) 
    

    """ --- Other interesting wavefunctions ---
   
    # Gaussian wave packet with momentum
    # Psi(x) = 1/(2πγ²)^(1/4) * exp(-0.5 * ((x - x0)/γ)²) * exp(i * k0 * x)
    # return (1 / (2 * np.pi * gamma**2) ** 0.25) * np.exp(-0.5 * (x - psioffset) / gamma ** 2 + 1j * k_0 * x, dtype=complex) 
    """

def main():
    """
    Runs the simulation and saves the results to an .npz file in the 'simulations_data' folder.
    The file is named according to the set parameters. It checks if a file with the same 
    parameters already exists and asks the user for confirmation before overwriting 
    (useful when testing different grids or resolutions).
    """
    output_folder = "simulations_data"
    os.makedirs(output_folder, exist_ok=True)
    
    pot_func = V_harmonic
    
    """ Case: Morse Potential 
    pot_func = V_morse
    pot_params = {'D_e': 50, 'a': 0.5, 'voffset': 0.0}
    """

    par = config

    # Merge function name and params for filename generation
    info_dict = {'pot_func': pot_func.__name__, **pot_params, **psi_params}

    # Generate filename string
    filename_str = "_".join([f"{key}-{value}" for key, value in info_dict.items()])
    filename_base = filename_str + ".npz"
    full_path = os.path.join(output_folder, filename_base)

    if os.path.exists(full_path):
        print(f"The file already exists: {filename_base}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled by the user.")
            return

    print("Initializing simulation...")
    
    # Lambda functions to wrap parameters, ensuring uniform call signature
    actual_V = lambda x, p: pot_func(x, p, **pot_params)
    actual_psi = lambda x, p: wavefunc(x, p, **psi_params)
    
    actual_frames = par.timesteps // par.steps_per_frame
    
    opr = init(par, V_func=actual_V, psi_func=actual_psi)
    
    run_save_simulation(
        par, opr, 
        filename=full_path, 
        n_frames=actual_frames, 
        steps_per_frame=par.steps_per_frame
    )
    
    print(f"Simulation completed, data saved to {full_path}.")

if __name__ == "__main__":
    main()