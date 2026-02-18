import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from split import frequency_correlation_from_ct 
from main import config
from scipy.special import hermite
from math import factorial, sqrt

""" 
This module post-processes all simulation data, creating a plot of the time correlation
and the Fourier transform of the correlation. It also identifies the peaks of the 
spectrum obtained from the potential function.
"""

# --- File Management ---
# Automatically detects the most recent simulation file for convenience.
folder = "simulations_data" 
file_name = "" # Set a specific filename here to override auto-detection

if file_name:
    open_file = os.path.join(folder, file_name)
else:
    open_file = None

def latest_file():
    # Finds the most recently modified .npz file in the data folder.
    if not os.path.exists(folder): return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
    if not files: return None
    return max(files, key=os.path.getctime)

def load_data(filename):
    # Loads simulation
    if not filename: filename = latest_file()
    if not filename or not os.path.exists(filename):
        print(f"File not found: {filename}")
        exit()
    return np.load(filename), filename

# --- Analytical Benchmarking (Optional) ---
""" 
The following block calculates analytical solutions for the Harmonic Oscillator.
It is useful for benchmarking the numerical precision of the Split Operator Method.
By comparing the numerical peaks with analytical eigenvalues (E_n = hbar*w*(n+0.5)),
we can validate the grid resolution and time step choices.
"""

"""
def eigenfunc_harmonic(x, par, n_max):
    phi_n_list = []
    mw_h = par.m * par.omega / par.hbar
    factor = (mw_h / np.pi) ** 0.25 
    
    for n in range(n_max + 1):
        H_n = hermite(n)
        norm_factor = factor / sqrt(2**n * factorial(n))
        phi_n = norm_factor * H_n(np.sqrt(mw_h) * x) * np.exp(-0.5 * mw_h * x ** 2)
        phi_n_list.append(phi_n)
    
    return phi_n_list

def eigenvalues_harmonic(freq_array, par, n_max):
    E_n_list = []
    w_max = np.max(freq_array)
    n_max = int(w_max / par.omega)

    for n in range(n_max + 1):
        E_n = par.hbar * par.omega * (n + 0.5)
        E_n_list.append(E_n)
    return E_n_list, n_max

def analytical_spectrum(freq, x, E_n_list, n_max, par, psi0):
    phi_n_list = eigenfunc_harmonic(x, par, n_max)
    Sw_analytical = np.zeros_like(freq, dtype=float)

    dx = x[1] - x[0]
    eta = freq[1] - freq[0]

    for n in range(n_max + 1):
        phi_n = phi_n_list[n]
        c_n = np.sum(np.conj(phi_n) * psi0) * dx
        w_n_center = E_n_list[n] / par.hbar

        diff_w = freq - w_n_center
        delta = (1 / np.pi) * (eta / (diff_w**2 + eta**2))

        Sw_analytical += np.abs(c_n)**2 * delta 

    return Sw_analytical
"""

# --- Visualization Setup ---
"""
Configures the static elements of the plot.
Uses a dual-axis system (twinx) to plot the Potential/Probability Density (Energy scale)
simultaneously with the Wavefunction's Real/Imaginary parts (Amplitude scale).
"""

def setup_plot(fig, ax1, x, psi_ref, V_ref, title="", xlims=None, ylims=None):
    density0 = np.abs(psi_ref[0])**2
    V_real = V_ref.real

    # Primary Axis (Left): Potential V(x) and Probability Density |Psi|^2
    line_V, = ax1.plot(x, V_real, color="black", lw=1.5, label="V(x)")
    line_density, = ax1.plot(x, density0, color="blue", lw=1.5, label=r"$|\psi|^2$")

    ax1.set_xlabel("x [a.u.]")
    ax1.set_ylabel("Energy / Probability Density", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.set_title(title)

    if xlims: ax1.set_xlim(xlims)
    if ylims: ax1.set_ylim(ylims)

    # Secondary Axis (Right): Real and Imaginary parts of Psi
    # This is necessary because Psi amplitude can be small compared to V(x) energy values.
    ax2 = ax1.twinx()
    y_max = np.max(np.abs(psi_ref)) if np.max(np.abs(psi_ref)) > 0 else 1
    
    line_real, = ax2.plot(x, psi_ref[0].real, color="orange", lw=1.0, linestyle=":", label=r"Re$\{\psi\}$")
    line_imag, = ax2.plot(x, psi_ref[0].imag, color="cyan", lw=1.0, linestyle=":", label=r"Im$\{\psi\}$")
    
    ax2.set_ylabel(r"Amplitude $\psi(x)$", color="black")
    ax2.set_ylim(-1.5*y_max, 1.5*y_max) 

    # Combine legends from both axes
    lines = [line_density, line_V, line_real, line_imag]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    return (line_density, line_real, line_imag), ax2

""" Main analysis function. 
1. Reconstructs the full time-correlation function.
2. Computes the Power Spectrum via FFT.
3. Identifies energy eigenvalues (peaks).
"""

def plot_simulation(data):
    c_t = data['correlation_history']
    dt = float(data['dt'])
    x = data['x']
    psi0 = data['psi_snapshots'][0]
    times = np.arange(len(c_t)) * dt

    par = config

    # --- Signal Processing Step ---
    # Construct the symmetric correlation function C_full.
    # The simulation gives us C(t) for t >= 0.
    # We use the property C(-t) = C*(t) to extend it to negative times.
    # This doubles the time window, improving spectral resolution (narrower peaks).
    total_steps = len(c_t)
    c_full = np.empty(2 * total_steps - 1, dtype=complex)
    
    # Fill positive times (center to right)
    c_full[total_steps - 1] = c_t[0]
    for i in range(1, total_steps):
        c_full[total_steps - 1 + i] = c_t[i]
        c_full[total_steps - 1 - i] = np.conj(c_t[i]) # Negative time mirror

    # Compute FFT to get the Energy Spectrum
    freq, Cw = frequency_correlation_from_ct(par, c_full)

    # --- Analytical Benchmark (Optional) ---
    # Uncomment the lines below to compare with the analytical Harmonic Oscillator solution
    # E_n_list, n_max = eigenvalues_harmonic(freq, par, n_max = None)
    # Sw_analytical = analytical_spectrum(freq, x, E_n_list, n_max, par, psi0)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

    # Subplot 1: Time Domain (Autocorrelation)
    ax1.plot(times, np.abs(c_t)**2, 'k-', lw=2, label=r"$|\sigma(t)|^2$")
    ax1.plot(times, c_t.real, label=r"Re{$\sigma$(t)}", alpha=0.7)
    ax1.plot(times, c_t.imag, label=r"Im{$\sigma$(t)}", alpha=0.7)
    ax1.set_title(r"Time Autocorrelation Function <$\psi$(0)|$\psi$(t)>")
    ax1.set_xlabel("Time [a.u.]")
    ax1.set_ylabel("Correlation Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Frequency Domain (Energy Spectrum)
    threshold = 0.1 # Minimum amplitude to be considered a peak

    ax2.plot(freq, np.abs(Cw), 'k-', lw=2, label=r"$|\sigma(\omega)|$ Power Spectrum")
    ax2.plot(freq, Cw.real, label=r"Re{$\sigma$(ω)}")
    ax2.plot(freq, Cw.imag, label=r"Im{$\sigma$(ω)}")
    ax2.set_title("Energy Spectrum (Fourier Transform of Correlation)")
    ax2.set_xlabel(r"Frequency $\omega$ (Energy/$\hbar$)")
    ax2.set_ylabel(r"Spectral Density $|S(\omega)|$")
    
    max_amp = np.max(np.abs(Cw))
    ax2.set_ylim(-max_amp*1.1, max_amp*1.1)
    ax2.set_xlim(-1, 100) # Limit x-axis to relevant energy range
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- Peak Identification Algorithm ---
    # Finds local maxima where the amplitude is greater than neighbors and threshold.
    Cw_abs = np.abs(Cw)
    center = Cw_abs[1:-1]
    left = Cw_abs[:-2]
    right = Cw_abs[2:]
    
    # Boolean mask for peaks
    peaks = (center > left) & (center > right) & (center > threshold)
    peak_indices = np.where(peaks)[0] + 1 # +1 to correct for slicing offset

    print("\n" + "="*40)
    print(f"SPECTRUM ANALYSIS | Identified Peaks: {len(peak_indices)}")
    print("="*40)
    if len(peak_indices) > 0:
        for idx in peak_indices:
            print(f"Mode Energy: {freq[idx]:.8f} | Amplitude: {Cw_abs[idx]:.4e}")
    print("="*40 + "\n")

    plt.tight_layout()
    plt.show()

def animate_wavefunction(data, original_filename, fps, xlims=None, ylims=None):
    #Generates an MP4 animation of the wavefunction evolution.
    #Uses 'blit=True' for performance optimization (only redrawing changed pixels).
    psi_snapshots = data['psi_snapshots']
    V = data['V']
    x = data['x']
    
    output_name = original_filename.replace('.npz', '_animation.mp4')
    print(f"Rendering video to: {output_name}")

    fig, ax1 = plt.subplots(figsize=(16, 9))
    (line_prob, line_real, line_imag), ax2 = setup_plot(
        fig, ax1, x, psi_snapshots, V, 
        title="Wavefunction Time Evolution", xlims=xlims, ylims=ylims
    )

    def update(frame):
        """Update function called for every frame of the animation."""
        psi = psi_snapshots[frame]
        line_prob.set_ydata(np.abs(psi)**2) 
        line_real.set_ydata(psi.real)
        line_imag.set_ydata(psi.imag)
        return line_prob, line_real, line_imag
    
    ani = FuncAnimation(fig, update, frames=len(psi_snapshots), blit=True)
    
    # Using FFMpegWriter for high-quality MP4 output
    try:
        writer = FFMpegWriter(fps=fps, bitrate=12000)
        ani.save(output_name, writer=writer, dpi=200)
        print("Video rendering finished successfully.")
    except Exception as e:
        print(f"Error saving video. Ensure FFMPEG is installed. Details: {e}")

if __name__ == "__main__":
    data, original_filename = load_data(open_file)
    
    # 1. Plot static analysis (Correlation & Spectrum)
    plot_simulation(data)
    
    # 2. Option to render video
    resp = input("Render video animation? (y/n): ")
    if resp.lower() == 'y':
        # Adjust these limits based on your potential well width and energy
        xlims = (-5, 5) 
        ylims = (-1, 20)
        animate_wavefunction(data, original_filename, fps=60, xlims=xlims, ylims=ylims)