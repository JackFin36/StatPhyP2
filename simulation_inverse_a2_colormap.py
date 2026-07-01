import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

k_B = 1
NUM_STATES = 3

T_BATH = 10
STATE_ENERGIES = np.array([0.0, 0.1, 1.0])
B12 = 2.0
B13 = 1.01
B23 = 10.0

T_VALUES = np.linspace(1e-3, 2, 200)
B_VALUES = np.linspace(0.0, 15.0, 200)


def define_boltzmann_distribution(T):
    p = np.exp(-STATE_ENERGIES / (k_B * T))
    p /= np.sum(p)
    return p


def define_transition_rate_matrix(T, B):
    energy_barriers = np.array([B12, B, B23])

    B = np.zeros((NUM_STATES, NUM_STATES))
    iu = np.triu_indices(NUM_STATES, k=1)
    B[iu] = energy_barriers
    B[(iu[1], iu[0])] = energy_barriers

    R = np.exp(-(B - STATE_ENERGIES[np.newaxis, :]) / (k_B * T))
    np.fill_diagonal(R, 0.0)
    R[np.diag_indices(NUM_STATES)] = -np.sum(R, axis=0)

    evals, vl, vr = eig(R, left=True, right=True)
    idx = np.argsort(np.real(evals))[::-1]
    evals = evals[idx]
    vl = vl[:, idx]
    vr = vr[:, idx]

    return R, evals, vl, vr


def compute_a2(T, B):
    _, _, vl, _ = define_transition_rate_matrix(T_BATH, B)
    p_eq = define_boltzmann_distribution(T_BATH)
    p = define_boltzmann_distribution(T)
    dp = p - p_eq
    a = vl.conj().T @ dp
    return np.abs(a[1])


a2_values = np.zeros((len(B_VALUES), len(T_VALUES)))

for i, B in enumerate(B_VALUES):
    for j, T in enumerate(T_VALUES):
        a2_values[i, j] = compute_a2(T, B)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.pcolormesh(T_VALUES, B_VALUES, a2_values, shading='auto', cmap='viridis')
ax.set_xlabel('T')
ax.set_ylabel(r'$B_{13}$')
ax.set_title(r'Colormap of $|a_2|$ for the inverse Mpemba setup')
fig.colorbar(im, ax=ax, label=r'$|a_2|$')
plt.tight_layout()
plt.show()
