import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os
from split import frequency_correlation_from_ct 
from main import config
from scipy.special import hermite
from math import factorial, sqrt

file_name = ""
folder = "simulations_data" 

if file_name:
    open_file = os.path.join(folder, file_name)
else:
    open_file = None

def latest_file():
    if not os.path.exists(folder): return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')]
    if not files: return None
    return max(files, key=os.path.getctime)

def load_data(filename):
    if not filename: filename = latest_file()
    if not filename or not os.path.exists(filename):
        print(f"File not found: {filename}")
        exit()
    return np.load(filename), filename

""" 
This part calculate the analytical eigenvalues and coefficients for the plot of a analytical
peaks of the Harmonic Oscilator in the frequency space. How we know exactly the eigenfunctions
and eigenvalues for this example, this makes him perfect to compare with the results obtained
in the Split Operathor method, evaluating the analytical and aproximated eigenenergys and
comparing, besides the height of each peak, that show us the influence of each eigenvalue
in the temporal evolution.

Here I calculate the eigenfunction of the HO and with this made a array with every eigenenergy.
and then the function analytical_spectrum uses a Lorentzian function to evaluate a peak and 
his location in the frequency space. Since we have Dirac-Delta function por each peak
we took the factor "η" in terms of the grid in the frequency space. 

"""

def eigenfunc_harmonic(x, par, n_max):
    phi_n_list = []
    mw_h = par.m * par.omega /par.hbar
    factor = (mw_h / np.pi) ** 0.25 
    
    for n in range(n_max + 1):
        H_n = hermite(n)
        norm_factor = factor  / sqrt(2**n *factorial(n))
        phi_n = norm_factor * H_n(np.sqrt (mw_h)* x) * np.exp(-0.5 * mw_h * x **2)
        phi_n_list.append(phi_n)
    
    return phi_n_list


def eigenvalues_harmonic(freq_array, par, n_max):
    E_n_list = []
    w_max = np.max(freq_array)
    n_max = int(w_max / par.omega)

    for n in range(n_max + 1):
        E_n = par.hbar * par.omega * (n + 0.5)
        E_n_list.append(E_n)
    return E_n_list , n_max

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
        delta =  (1 / np.pi ) * (eta / (diff_w**2 + eta**2))

        Sw_analytical += np.abs(c_n)**2 * delta 
    print(np.sum(Sw_analytical)* eta)
    return Sw_analytical


""" 
Here I just plot the graphs and simulation video of the temporal evolution.
"""

def setup_plot(fig, ax1, x, psi_ref, V_ref, title="", xlims = None, ylims = None):
    density0 = np.abs(psi_ref[0])**2
    V_real = V_ref.real

    line_V, = ax1.plot(x, V_real, color="black", lw=1.5, label="V(x)")
    line_density, = ax1.plot(x, density0, color="blue", lw=1.5, label=r"$|\psi|^2$")

    ax1.set_xlabel("x")
    ax1.set_ylabel("Energy", color="black")
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.set_title(title)

    if xlims: ax1.set_xlim(xlims)
    if ylims: ax1.set_ylim(ylims)

    ax2 = ax1.twinx()
    y_max = np.max(np.abs(psi_ref)) if np.max(np.abs(psi_ref)) > 0 else 1
    
    line_real, = ax2.plot(x, psi_ref[0].real, color="orange", lw=1.0, linestyle=":", label=r"Re$\{\psi\}$")
    line_imag, = ax2.plot(x, psi_ref[0].imag, color="cyan", lw=1.0, linestyle=":", label=r"Im$\{\psi\}$")
    
    ax2.set_ylabel(r"Re/Im $\psi(x)$", color="black")
    ax2.set_ylim(-1.5*y_max, 1.5*y_max)

    lines = [line_density, line_V, line_real, line_imag]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    return (line_density, line_real, line_imag), ax2


def plot_simulation(data):
    c_t = data['correlation_history']
    dt = float(data['dt'])
    x = data['x']
    psi0 = data['psi_snapshots'][0]
    times = np.arange(len(c_t)) * dt

    par = config

    total_steps = len(c_t)
    c_full = np.empty(2 * total_steps - 1, dtype=complex)
    c_full[total_steps - 1] = c_t[0]
    for i in range (1, total_steps):
        c_full[total_steps - 1 + i] = c_t[i]
        c_full[total_steps - 1 - i] = np.conj(c_t[i])

    dummy_par = type('Params', (object,), {'dt': dt , 'hbar': par.hbar})()
    freq, Cw = frequency_correlation_from_ct(dummy_par, c_full)
    d_omega = freq[1] - freq[0]

    print( np.sum(Cw.real)*d_omega)

    E_n_list, n_max = eigenvalues_harmonic(freq, par, n_max = None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

    # Graph 1: Time Correlation
    ax1.plot(times, np.abs(c_t)**2, 'k-', lw=2, label=r"$|\sigma(t)|^2$")
    ax1.plot(times, c_t.real, label=r"Re{$\sigma$(t)}")
    ax1.plot(times, c_t.imag, label=r"Im{$\sigma$(t)}")
    ax1.set_title("Time Correlation Function")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"$| \sigma (t)|^2$")
    ax1.legend()
    ax1.grid()
    
    Sw_analytical = analytical_spectrum(freq, x, E_n_list, n_max, par, psi0)

    threshold = 0.1

    ax2.plot(freq, np.abs(Cw), 'k-', lw=2, label=r"$|\sigma(\omega)|$")
    ax2.plot(freq, Sw_analytical, 'r--', lw=1.5, label="Analytical Spectrum")
    ax2.plot(freq, Cw.real, label=r"Re{$\sigma$(ω)}")
    ax2.plot(freq, Cw.imag, label=r"Im{$\sigma$(ω)}")
    ax2.set_title("Frequency Correlation Function")
    ax2.set_xlabel("Frequency (ω)")
    ax2.set_ylabel(r"$|S(\omega)|$")
    
    max_amp = np.max(np.abs(Cw))
    ax2.set_ylim(-max_amp*1.1, max_amp*1.1)
    ax2.set_xlim(-1, np.max(freq)) 
    ax2.grid()
    ax2.legend()

    # Peaks identify
    Cw = np.abs(Cw)
    center = Cw[1:-1]
    left = Cw[:-2]
    right = Cw[2:]
    peaks = (center > left) & (center > right) & (center > threshold)
    peak_indices = np.where(peaks)[0] + 1

    print(f"Identified Peaks: (Total: {len(peak_indices)})")
    if len(peak_indices) > 0:
        for idx in peak_indices:
            print(f"Energy: {freq[idx]:.8f}, Amplitude: {Cw[idx]:.4e}")

    plt.tight_layout()
    plt.show()

def animate_wavefunction(data, original_filename, fps, xlims=None, ylims=None):
    psi_snapshots = data['psi_snapshots']
    V = data['V']
    x = data['x']
    
    output_name = original_filename.replace('.npz', '_animation.mp4')

    fig, ax1 = plt.subplots(figsize=(16, 9))
    (line_prob, line_real, line_imag), ax2 = setup_plot(
        fig, ax1, x, psi_snapshots, V, 
        title="Wavefunction Evolution", xlims=xlims, ylims=ylims
    )

    def update(frame):
        psi = psi_snapshots[frame]
        line_prob.set_ydata(np.abs(psi)**2) 
        line_real.set_ydata(psi.real)
        line_imag.set_ydata(psi.imag)
        return line_prob, line_real, line_imag
    
    ani = FuncAnimation(fig, update, frames=len(psi_snapshots), blit=True)
    
    writer = FFMpegWriter(fps=fps, bitrate=12000)
    ani.save(output_name, writer=writer, dpi=200)
    print(f"Saved video: {output_name}")

if __name__ == "__main__":
    data, original_filename = load_data(open_file)
    
    plot_simulation(data)
    
    resp = input("Render video? (y/n): ")
    if resp.lower() == 'y':
        xlims = (-1, 1)
        ylims = (-1, 20)
        animate_wavefunction(data, original_filename, fps=60, xlims=xlims, ylims=ylims)