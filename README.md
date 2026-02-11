# Quantum Wavefunction Simulation

This repository focuses on simulating the dynamics of an arbitrary wavefunction in a given potential using a numerical approach to solve the Schrödinger Equation.

The time-dependent Schrödinger equation is given by:

$$
i \hbar \frac{\partial}{\partial t} |\psi\rangle = \left( -\frac{\hbar^2}{2m} \nabla^2 + \hat{V} \right) |\psi\rangle
$$

Or, in terms of position and time:

$$
i\hbar \frac{\partial}{\partial t} \psi(r,t) = -\frac{\hbar^2}{2m} \nabla^2 \psi(r,t) + V(r,t) \psi(r,t)
$$

The time evolution of the wavefunction is defined formally as:

$$
\psi(r, t+dt) = e^{-\frac{i \hat{H} dt}{\hbar}} \psi(r,t)
$$

Since the kinetic energy operator ($\hat{T}$) does not commute with the potential energy operator ($\hat{V}$), we apply the **Split-Operator method** (Strang splitting):

$$
\psi(t+dt) \approx e^{-\frac{i\hat{V}dt}{2\hbar}} e^{-\frac{i\hat{T}dt}{\hbar}} e^{-\frac{i\hat{V}dt}{2\hbar}} \psi(t) + \mathcal{O}(dt^3)
$$

## Code Structure

The code is composed primarily of three parts: `split.py`, `main.py`, and `viewer.py`.

### `main.py`
Sets the simulation parameters, including:
* Spatial grid and resolution.
* Timesteps and time resolution.
* Physical constants ($\hbar$, mass, etc.).
* The initial wavefunction and the potential function.

It runs the logic in `split.py` and saves the raw data of the evolution package.

### `split.py`
Contains the four principal functions needed for the analysis of the simulation:
1.  **`step`**: Propagates the wavefunction using the **Fast Fourier Transform (FFT)** to toggle between position and momentum space, allowing the application of the Kinetic and Potential operators in their respective diagonal bases.
2.  **`time_correlation`**: Calculates the autocorrelation $\langle \psi(0) | \psi(t) \rangle$ at each time step.
3.  **Spectrum calculation**: Functions to compute the energy spectrum via the Fourier transform of the time correlation.

All principal values are saved in a `.npz` file that compiles the wavefunction snapshots (in specific frames), correlation history, potential, spatial grid, timesteps, and the total energy expectation value.

### `viewer.py`
Loads the saved data to generate:
* Plots of the time correlation.
* The Fourier transform of the correlation (Energy Spectrum).
* An animation of the wavefunction evolving over time.

## Objective

The ultimate goal is to use the split-operator method and other techniques to study the absorption and emission of electrons within **RIXS theory** (Resonant Inelastic X-ray Scattering).