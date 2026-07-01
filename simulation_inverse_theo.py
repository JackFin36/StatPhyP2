import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

k_B = 1

NUM_STATES = 3
TIME_STEP_SIZE = 1e-3
FINAL_TIME = 2
NUM_TIME_STEPS = int(FINAL_TIME / TIME_STEP_SIZE)

T_HOT = 0.26
T_COLD = 0.08
T_BATH = 10

STATE_ENERGIES = np.array([0.0, 0.1, 1.0])
ENERGY_BARIERS = np.array([2.0, 1.01, 10.0])

print("State energies:", STATE_ENERGIES)
print("State bariers:", ENERGY_BARIERS)

def define_Boltzmann_distribution(T):
    Boltzmann_distribution = np.exp(-STATE_ENERGIES / (k_B * T))
    Boltzmann_distribution /= np.sum(Boltzmann_distribution)
    return Boltzmann_distribution

def define_transition_rate_matrix(T, energy_barriers=ENERGY_BARIERS):
    # Build symmetric barrier matrix B
    B = np.zeros((NUM_STATES, NUM_STATES))
    iu = np.triu_indices(NUM_STATES, k=1)
    B[iu] = energy_barriers
    B[(iu[1], iu[0])] = energy_barriers  # mirror to lower triangle

    # Off-diagonal rates: R_ij = exp(-(B_ij - E_j)/(k_B T))
    R = np.exp(-(B - STATE_ENERGIES[np.newaxis, :]) / (k_B * T))

    # No self-transitions
    np.fill_diagonal(R, 0.0)

    # Set diagonal so each column sums to zero
    R[np.diag_indices(NUM_STATES)] = -np.sum(R, axis=0)

    evals, vl, vr = eig(R, left=True, right=True)

    return R, evals, vl, vr

def evolve_distribution(p, R):
    d = np.zeros((NUM_TIME_STEPS, NUM_STATES))
    for i in range(NUM_TIME_STEPS):
        d[i] = p
        p += (R @ p) * TIME_STEP_SIZE
    return d

def distance(p, T):
    equilibrium_distribution = define_Boltzmann_distribution(T)

    D = STATE_ENERGIES * (p - equilibrium_distribution) / (k_B * T)
    D += p * np.log(p)
    D -= equilibrium_distribution * np.log(equilibrium_distribution)

    return np.sum(D, axis=1)


temperatures = np.linspace(1e-3, 2, 100)
times = np.linspace(0, FINAL_TIME, NUM_TIME_STEPS)
distances = np.zeros((NUM_TIME_STEPS, len(temperatures)))

R, evals, vl, vr = define_transition_rate_matrix(T_BATH)
idx = np.argsort(np.real(evals))[::-1]
evals = evals[idx]
vl = vl[:, idx]
vr = vr[:, idx]

print("Transition rate matrix R:\n", R)
print("Eigenvalues:", evals)
print("Left eigenvectors:\n", vl)
print("Right eigenvectors:\n", vr)

p_eq = define_Boltzmann_distribution(T_BATH)
a = np.zeros((len(temperatures), NUM_STATES), dtype=complex)
for i, T in enumerate(temperatures):
    print("Temperature T =", T)
    p = define_Boltzmann_distribution(T)
    print("Initial Boltzmann distribution p =", p)
    dp = p - p_eq
    a_i = vl.conj().T @ dp
    print("a_i:", a_i)
    a[i] = a_i
    s = evolve_distribution(p, R)
    distances[:, i] = distance(s, T_BATH)

fig = plt.figure(figsize=(14, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)

ax1.plot(temperatures, np.abs(a[:, 1]), label='|a2|')
ax1.axvline(T_HOT, color='tab:orange', linestyle='--', alpha=0.7, label=f'T_hot={T_HOT:.2f}')
ax1.axvline(T_COLD, color='tab:blue', linestyle='--', alpha=0.7, label=f'T_cold={T_COLD:.2f}')
ax1.set_xlabel('Initial temperature')
ax1.set_ylabel('|a2|')
ax1.set_title('Slow-mode coefficient')
ax1.legend()

for T in [T_HOT, T_COLD]:
    print("Temperature T =", T)
    p = define_Boltzmann_distribution(T)
    print("Initial Boltzmann distribution p =", p)
    print("a_i:", vl.conj().T @ (p - p_eq))
    s = evolve_distribution(p, R)
    ax2.plot(times, distance(s, T_BATH), label=f"T={T:.2f}")
ax2.set_xlabel('Time')
ax2.set_ylabel('Distance to equilibrium')
ax2.set_title('Relaxation curves')
ax2.legend()

B13_values = np.linspace(0.0, 15.0, 200)
a2_map = np.zeros((len(B13_values), len(temperatures)))

for i, B13 in enumerate(B13_values):
    barriers = np.array([ENERGY_BARIERS[0], B13, ENERGY_BARIERS[2]])
    _, evals_B, vl_B, vr_B = define_transition_rate_matrix(T_BATH, barriers)
    idx_B = np.argsort(np.real(evals_B))[::-1]
    vl_B = vl_B[:, idx_B]
    vr_B = vr_B[:, idx_B]
    for k in range(NUM_STATES):
        norm = vl_B[:, k].conj().T @ vr_B[:, k]
        vl_B[:, k] /= norm

    for j, T in enumerate(temperatures):
        p = define_Boltzmann_distribution(T)
        dp = p - p_eq
        a2_map[i, j] = np.abs((vl_B.conj().T @ dp)[1])

im = ax3.pcolormesh(temperatures, B13_values, a2_map, shading='auto', cmap='viridis')
ax3.scatter([T_HOT], [ENERGY_BARIERS[1]], color='red', label='chosen parameters')
ax3.scatter([T_COLD], [ENERGY_BARIERS[1]], color='red')
ax3.set_xlabel('T')
ax3.set_ylabel(r'$B_{13}$')
ax3.set_title(r'Colormap of $|a_2|$')
ax3.legend()
fig.colorbar(im, ax=ax3, label=r'$|a_2|$')

fig.tight_layout()
plt.show()

