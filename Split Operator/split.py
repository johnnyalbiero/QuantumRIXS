import numpy as np
import math
from operators import Parameters, Operators


# This init function initializes the Operators class with the potential and initial wave function, as well as the kinetic and potential evolution operators
def init(par: Parameters, V_func, psi_func):
    opr = Operators(len(par.x))
    opr.V = V_func(par.x)
    opr.psi = psi_func(par.x).astype(complex)

    if par.im_time:
        opr.K = np.exp(-0.5 * par.dt * (par.hbar * par.k ** 2)/par.m)
        opr.R = np.exp(-0.5 * par.dt * opr.V / par.hbar)
    else:
        opr.K = np.exp(-0.5j * par.dt * (par.hbar * par.k ** 2)/par.m)
        opr.R = np.exp(-0.5j * par.dt * opr.V / par.hbar)
    return opr

# This split_operator function does the full time evolution for par.timesteps, using a loop to call the step function for itch timestep
def split_operator(par: Parameters, opr: Operators):
    for i in range(par.timesteps):
        opr.psi *= opr.R
        opr.psi = np.fft.fft(opr.psi)
        opr.psi *= opr.K
        opr.psi = np.fft.ifft(opr.psi)
        opr.psi *= opr.R
        density = np.abs(opr.psi) ** 2

        if par.im_time:
            renorm_factor = np.sum(density) * par.dx
            opr.psi /= np.sqrt(renorm_factor)
    if i % (par.timesteps // 100) == 0:
        filename = f"output{str(i).rjust(5, '0')}.dat"
        with open(filename, "w") as outfile:
            for j in range(len(density)):
                line = f"{par.x[j]}\t{density[j].real}\t{opr.V[j].real}\n"
                outfile.write(line)
        print("Outputting step: ", i + 1)

# Tghe step function does a single time step of the split-operator method and returns the density, that's the function used in animation.py
# to update the wavefunction at each frame of the animation
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


# Cute function that calculate the total energy of the system at a given time t
def calculate_energy(par: Parameters, opr: Operators):
    psi_r = opr.psi
    psi_k = np.fft.fft(psi_r)
    psi_c = np.conj(psi_r)

 # T_op = (hbar² * k²) / 2m
    kinetic_operator_k_space = (par.hbar**2 * par.k**2) / (2 * par.m)
    energy_k = psi_c * np.fft.ifft(kinetic_operator_k_space * psi_k)
    energy_r = psi_c * opr.V * psi_r

    total_energy = np.sum(energy_k + energy_r) * par.dx
    return total_energy.real