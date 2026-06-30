import numpy as np
import matplotlib.pyplot as plt


k_B = 1

NUM_STATES = 3
NUM_TIME_STEPS = int(1e5)
TIME_STEP_SIZE = 1e-3

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

    D = STATE_ENERGIES * (p - equilibrium_distribution) / T
    D += p * np.log(p)
    D -= equilibrium_distribution * np.log(equilibrium_distribution)

    return np.sum(D, axis=1)


R = define_transition_rate_matrix(T_BATH)
print(f"Transition rate matrix R =\n{R}")

for T in [T_HOT, T_COLD]:
    print("Temperature T =", T)
    
    p = define_Boltzmann_distribution(T)
    print("Initial Boltzmann distribution p =", p)

    s = evolve_distribution(p, R)
    plt.plot(distance(s, T_BATH), label=f"T = {T}")

plt.legend()
plt.show()

