import numpy as np
from split import init, Operators, Parameters
from animation import animate_simulation

# Funções de potencial e função de onda
def potencial(x, voffset: float = 0):
    return 0.5 * (x - voffset) ** 2  # Potencial harmônico 1/2 (x - x0)^2
    #return np.piecewise(x, [x < -1, (x >= -1) & (x <= 1), x > 1], [0, 5, 0])

def wavefunc(x, psioffset: float = -6):
    return (1 / (2 * np.pi) ** 0.25) * np.exp(-0.5 * (x - psioffset) ** 2, dtype=complex) # Psi(x) (1/(2pi)^(1/4)) * exp(- (x-x0)^2 / 2 )
    #return (1 / (2 * np.pi) ** 0.25) * np.exp(-0.5 * (x - psioffset) ** 2 + 1j * 5 * x, dtype=complex) # Psi(x) com momento inicial

def main():
    # Parâmetros da simulação
    par = Parameters(xmax=10, res=516, dt=0.01, timesteps=1500)
    opr = init(par, V_func=lambda x: potencial(x), psi_func=lambda x: wavefunc(x))

    # Parâmetros da animação
    n_frames = 1500
    steps_per_frame = 1

    # Roda e salva a animação
    animate_simulation(par, opr, n_frames, steps_per_frame, dpi=200)

if __name__ == "__main__":
    main()