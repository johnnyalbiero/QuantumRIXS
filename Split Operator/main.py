import numpy as np
import os
from split import init, Parameters, run_save_simulation

config = Parameters(
    xmin=-5, xmax=5, res = 2**15, # How the algoritm use the FFT to calculate the dinamic of the wavefunction it's better to use 2^n to resolution
    dt=0.01, timesteps=2**18, steps_per_frame=30,
    hbar=1, m=1, omega=5
)
pot_params = {'voffset': 0.0}
psi_params = {'gamma': 0.1, 'k_0': -0.2, 'psioffset': 0.1}

def V_harmonic(x, par, voffset):
    return 0.5 * par.m * (par.omega**2) * (x - voffset)**2

def V_morse(x, D_e, a, voffset):
    return D_e * (1 - np.exp(-a * (x - voffset)))**2

    """ --- Interesting function ---
    lambda_ = np.sqrt(2 * par.m * D_e) / (a * par.hbar)
    n_max = int(lambda_ - 0.5)

    #for i in range(n_max + 1):
        #E_n = (par.hbar * a)**2 / (2 *par.m) * (lambda_**2 - (lambda_ - i - 0.5)**2)
        #print(f"Energy level n={i}: E_n = {E_n}")  
    """

def wavefunc(x, par, gamma, k_0, psioffset):
    # Gaussian wave packet
    #return (1 / (2 * np.pi * gamma**2) ** 0.25) * np.exp(-0.25 * ((x - psioffset) / gamma) ** 2) # Psi(x) = 1/(2πγ²)^(1/4) * exp(-0.5 * ((x - x0)/γ)²)
    
    # Ground state of quantum harmonic oscillator
    # Psi(x) = (mω/πħ)^(1/4) * exp(-0.5 * mω/ħ * (x - x0)²)
    return (par.omega * par.m / (np.pi * par.hbar)) ** 0.25 * np.exp(-0.5 * (par.m * par.omega /par.hbar) * (x - psioffset) ** 2) 
    

    """ --- Another interesting wave functions ---

   
    # Gaussian wave packet with momentum
    # Psi(x) = 1/(2πγ²)^(1/4) * exp(-0.5 * ((x - x0)/γ)²) * exp(i * k0 * x)
    #return (1 / (2 * np.pi * gamma**2) ** 0.25) * np.exp(-0.5 * (x - psioffset) / gamma ** 2 + 1j * k_0 * x, dtype=complex) 
    """
def main():
    output_folder = "simulations_data"
    os.makedirs(output_folder, exist_ok=True)
    
    pot_func = V_harmonic
    
    """ Other case: Morse Potential 
    pot_func = V_morse
    pot_params = {'D_e': 50, 'a': 0.5, 'voffset': 0.0}
    """

    par = config

    info_dict = {'pot_func': pot_func.__name__, **pot_params, **psi_params}

    filename_base = "_".join([f"{key}-{value}" for key, value in info_dict.items()]) + ".npz"
    full_path = os.path.join(output_folder, filename_base)

    if os.path.exists(full_path):
        print(f"The file already exists: {filename_base}.npz")
        resposta = input("Do you want to overwrite it? (y/n): ")
        if resposta.lower() != 'y':
            print("Operation cancelled by the user.")
            return

    print("Initializing simulation...")
    actual_V = lambda x, p: pot_func(x, p, **pot_params)
    actual_psi = lambda x, p: wavefunc(x, p, **psi_params)
    actual_frames = par.timesteps // par.steps_per_frame
    
    opr = init(par, V_func=actual_V, psi_func=actual_psi)
    run_save_simulation(par, opr, filename=full_path, n_frames=actual_frames, steps_per_frame=par.steps_per_frame)
    
    print(f"Simulation completed, data saved to {full_path}.")

if __name__ == "__main__":
    main()