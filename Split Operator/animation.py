import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import os
ffmpeg_path = os.environ.get("FFMPEG_PATH", r"ffmpeg")  # Adjust the path as necessary
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
from operators import Parameters, Operators
from split import step

def animate_simulation(par: Parameters, opr: Operators, n_frames, steps_per_frame,
 save_as: str = "Split_Operator.mp4", fps =60, dpi: int = 200):
    max_possible_frames = max(1, par.timesteps // max(1, steps_per_frame))
    n_frames = min(n_frames, max_possible_frames)

    fig, ax1 = plt.subplots(figsize=(16,8))

    density0 = np.abs(opr.psi)**2
    density_max = np.max(density0)

    line_density, = ax1.plot(par.x, density0, color="blue", lw=1.5, label=r"$|\psi(x)|^2$")

    V_scaled = opr.V.real / np.max(opr.V.real) * density_max
    line_V, = ax1.plot(par.x, V_scaled, color="black", lw=1.5, label="V(x)")

    ax1.set_xlim(par.x[0], par.x[-1])
    ax1.set_ylim(0, density_max * 1.5)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$|\psi(x)|^2$", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    y_max = np.max(np.abs(opr.psi.real))
    line_real, = ax2.plot(par.x, opr.psi.real, color="orange", lw=1.0, linestyle=":", label=r"Re$\{\psi(x)\}$")
    line_imag, = ax2.plot(par.x, opr.psi.imag, color="cyan", lw=1.0, linestyle=":", label=r"Im$\{\psi(x)\}$")
    ax2.set_ylim(-1.5*y_max, 1.5*y_max)
    ax2.set_ylabel(r"Re/Im $\psi(x)$", color="black")
    ax2.tick_params(axis='y', labelcolor="black")

    lines = [line_density, line_V, line_real, line_imag]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    def init():
        line_density.set_ydata(np.abs(opr.psi)**2)
        line_real.set_ydata(opr.psi.real)
        line_imag.set_ydata(opr.psi.imag)
        return line_density, line_V, line_real, line_imag

    last_frame = None

    def update(frame_idx):
        nonlocal last_frame
        if frame_idx != last_frame:
            if frame_idx > 0:
                for _ in range(steps_per_frame):
                    step(par, opr)
            last_frame = frame_idx

        line_density.set_ydata(np.abs(opr.psi)**2)
        line_real.set_ydata(opr.psi.real)
        line_imag.set_ydata(opr.psi.imag)
        return line_density, line_V, line_real, line_imag

    ani = FuncAnimation(fig, update, frames=range(n_frames), init_func=init, blit=True)

    # Salvar vídeo ou GIF
    if save_as.endswith(".mp4"):
            writer = FFMpegWriter(fps=fps)
            ani.save(save_as, writer=writer, dpi=dpi)
            print(f"Saved video: {save_as}")
    elif save_as.endswith(".gif"):
            writer = PillowWriter(fps=fps)
            ani.save(save_as, writer=writer, dpi=dpi)
    else:
        print("Unknown format. Use .mp4 or .gif")

    plt.close(fig)
