import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

k_B = 1

NUM_STATES = 3
NUM_TIME_STEPS = int(1e4)
TIME_STEP_SIZE = 1e-2

T_HOT = 4.25
T_COLD = 1.15
T_BATH = 0.05

STATE_ENERGIES = np.array([00., 0.1, 0.7])
ENERGY_BARIERS = np.array([1.5, 0.8, 1.2])

print("State energies:", STATE_ENERGIES)
print("State bariers:", ENERGY_BARIERS)

def define_Boltzmann_distribution(T):
    Boltzmann_distribution = np.exp(-STATE_ENERGIES / (k_B * T))
    Boltzmann_distribution /= np.sum(Boltzmann_distribution)
    return Boltzmann_distribution

def define_transition_rate_matrix(T):
    # Build symmetric barrier matrix B
    B = np.zeros((NUM_STATES, NUM_STATES))
    iu = np.triu_indices(NUM_STATES, k=1)
    B[iu] = ENERGY_BARIERS
    B[(iu[1], iu[0])] = ENERGY_BARIERS  # mirror to lower triangle

    # Off-diagonal rates: R_ij = exp(-(B_ij - E_j)/(k_B T))
    R = np.exp(-(B - STATE_ENERGIES[np.newaxis, :]) / (k_B * T))

    # No self-transitions
    np.fill_diagonal(R, 0.0)

    # Set diagonal so each column sums to zero
    R[np.diag_indices(NUM_STATES)] = -np.sum(R, axis=0)

    return R

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


ax = plt.figure().add_subplot(projection='3d')
temperatures = np.linspace(T_BATH, T_HOT, 30)
times = np.arange(NUM_TIME_STEPS) * TIME_STEP_SIZE
distances = np.zeros((NUM_TIME_STEPS, len(temperatures)))

R = define_transition_rate_matrix(T_BATH)
print(f"Transition rate matrix R =\n{R}")

for i, T in enumerate(temperatures):
    print("Temperature T =", T)
    
    p = define_Boltzmann_distribution(T)
    print("Initial Boltzmann distribution p =", p)

    s = evolve_distribution(p, R)
    distances[:, i] = distance(s, T_BATH)

X, Y = np.meshgrid(temperatures, times)
ax.plot_surface(X, Y, distances, cmap='PuBu', norm=LogNorm(vmin=T_BATH, vmax=0.3))
ax.set_xlabel('Initial temperature')
ax.set_ylabel('Time')
ax.set_zlabel('Distance to equilibrium')
plt.show()

