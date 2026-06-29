import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# --- 1. Statische Simulationsparameter ---
k_B = 1.0  
dt = 0.01
t_max = 20.0
times = np.arange(0, t_max, dt)

# --- 2. Hilfsfunktionen ---
def boltzmann_distribution(T, E):
    """Berechnet die Boltzmann-Verteilung in Abhängigkeit von T und E."""
    weights = np.exp(-np.array(E) / (k_B * T))
    return weights / np.sum(weights)

def transition_rate_matrix(T, E, B_matrix):
    """Berechnet die Ratenmatrix in Abhängigkeit von T, E und B."""
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                R[i, j] = np.exp(-(B_matrix[i, j] - E[j]) / (k_B * T))
    # Diagonalelemente anpassen
    for j in range(3):
        R[j, j] = -np.sum(R[:, j])
    return R

def kullback_leibler_divergence(p, pi):
    epsilon = 1e-12
    p_safe = np.clip(p, epsilon, 1.0)
    pi_safe = np.clip(pi, epsilon, 1.0)
    return np.sum(p_safe * np.log(p_safe / pi_safe))

def euclidean_distance(p, pi):
    return np.sqrt(np.sum((p - pi)**2))

# --- 3. Simulations-Logik ---
def run_simulation(Tc_val, Th_val, Tb_val, E_val, B_matrix):
    # Ziel-Gleichgewichtszustand (Bad)
    pi_b = boltzmann_distribution(Tb_val, E_val)
    R_b = transition_rate_matrix(Tb_val, E_val, B_matrix)
    
    # Startbedingungen berechnen
    p_cold = boltzmann_distribution(Tc_val, E_val)
    p_hot = boltzmann_distribution(Th_val, E_val)
    
    kl_cold, kl_hot = np.zeros_like(times), np.zeros_like(times)
    euc_cold, euc_hot = np.zeros_like(times), np.zeros_like(times)
    
    for i in range(len(times)):
        kl_cold[i] = kullback_leibler_divergence(p_cold, pi_b)
        kl_hot[i] = kullback_leibler_divergence(p_hot, pi_b)
        euc_cold[i] = euclidean_distance(p_cold, pi_b)
        euc_hot[i] = euclidean_distance(p_hot, pi_b)
        
        # Vorwärts-Euler Schritt
        p_cold = p_cold + np.dot(R_b, p_cold) * dt
        p_hot = p_hot + np.dot(R_b, p_hot) * dt
        
    return kl_cold, kl_hot, euc_cold, euc_hot

# --- 4. Plot Setup ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Platz machen für die Textboxen (3 Reihen unten)
plt.subplots_adjust(bottom=0.35)

# Startwerte initialisieren
init_E = [0.0, 0.1, 0.7]
init_B = np.zeros((3, 3))
init_B[0, 1] = init_B[1, 0] = 1.5
init_B[0, 2] = init_B[2, 0] = 0.8
init_B[1, 2] = init_B[2, 1] = 1.2
init_Tb, init_Tc, init_Th = 0.1, 0.42, 1.3

# Initiale Berechnung
kl_c, kl_h, euc_c, euc_h = run_simulation(init_Tc, init_Th, init_Tb, init_E, init_B)

# Linien plotten
line_kl_cold, = ax1.plot(times, kl_c, label=f'Kalt (T={init_Tc})', color='blue', linestyle='--')
line_kl_hot, = ax1.plot(times, kl_h, label=f'Heiss (T={init_Th})', color='red')
line_euc_cold, = ax2.plot(times, euc_c, label=f'Kalt (T={init_Tc})', color='blue', linestyle='--')
line_euc_hot, = ax2.plot(times, euc_h, label=f'Heiss (T={init_Th})', color='red')

for ax in [ax1, ax2]:
    ax.set_yscale('log')
    ax.set_xlabel('Zeit t')
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()

ax1.set_title('Kullback-Leibler Divergenz')
ax2.set_title('Euklidische Distanz')

# --- 5. Eingabefelder (TextBoxes) anlegen ---
# Format: [links, unten, breite, höhe]

# Reihe 1: Energien E
axbox_e1 = plt.axes([0.15, 0.22, 0.1, 0.04])
axbox_e2 = plt.axes([0.45, 0.22, 0.1, 0.04])
axbox_e3 = plt.axes([0.75, 0.22, 0.1, 0.04])

# Reihe 2: Barrieren B
axbox_b12 = plt.axes([0.15, 0.15, 0.1, 0.04])
axbox_b13 = plt.axes([0.45, 0.15, 0.1, 0.04])
axbox_b23 = plt.axes([0.75, 0.15, 0.1, 0.04])

# Reihe 3: Temperaturen T
axbox_tb = plt.axes([0.15, 0.08, 0.1, 0.04])
axbox_tc = plt.axes([0.45, 0.08, 0.1, 0.04])
axbox_th = plt.axes([0.75, 0.08, 0.1, 0.04])

# TextBoxes erstellen mit Startwerten
text_e1 = TextBox(axbox_e1, 'E1: ', initial='0.0')
text_e2 = TextBox(axbox_e2, 'E2: ', initial='0.1')
text_e3 = TextBox(axbox_e3, 'E3: ', initial='0.7')

text_b12 = TextBox(axbox_b12, 'B12: ', initial='1.5')
text_b13 = TextBox(axbox_b13, 'B13: ', initial='0.8')
text_b23 = TextBox(axbox_b23, 'B23: ', initial='1.2')

text_tb = TextBox(axbox_tb, 'Tb (Bad): ', initial='0.1')
text_tc = TextBox(axbox_tc, 'Tc (Kalt): ', initial='0.42')
text_th = TextBox(axbox_th, 'Th (Heiss): ', initial='1.3')

# --- 6. Update-Logik ---
def submit(expression):
    """Wird aufgerufen, wenn in einem Textfeld 'Enter' gedrückt wird."""
    try:
        # Aktuelle Werte aus den Textfeldern als Float auslesen
        e_val = [float(text_e1.text), float(text_e2.text), float(text_e3.text)]
        
        b_matrix = np.zeros((3, 3))
        b_matrix[0, 1] = b_matrix[1, 0] = float(text_b12.text)
        b_matrix[0, 2] = b_matrix[2, 0] = float(text_b13.text)
        b_matrix[1, 2] = b_matrix[2, 1] = float(text_b23.text)
        
        tb_val = float(text_tb.text)
        tc_val = float(text_tc.text)
        th_val = float(text_th.text)
        
        # Neue Simulation starten
        kl_c, kl_h, euc_c, euc_h = run_simulation(tc_val, th_val, tb_val, e_val, b_matrix)
        
        # Liniendaten updaten
        line_kl_cold.set_ydata(kl_c)
        line_kl_hot.set_ydata(kl_h)
        line_euc_cold.set_ydata(euc_c)
        line_euc_hot.set_ydata(euc_h)
        
        # Legenden updaten
        line_kl_cold.set_label(f'Kalt (T={tc_val})')
        line_kl_hot.set_label(f'Heiss (T={th_val})')
        line_euc_cold.set_label(f'Kalt (T={tc_val})')
        line_euc_hot.set_label(f'Heiss (T={th_val})')
        ax1.legend()
        ax2.legend()
        
        # Achsenskalierung anpassen
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        
        # Zeichnen
        fig.canvas.draw_idle()
        
    except ValueError:
        # Fehler abfangen, falls z.B. Buchstaben eingegeben werden
        print("Ungültige Eingabe. Bitte nur Zahlen verwenden.")

# Event-Listener an alle Textboxen binden
text_e1.on_submit(submit)
text_e2.on_submit(submit)
text_e3.on_submit(submit)
text_b12.on_submit(submit)
text_b13.on_submit(submit)
text_b23.on_submit(submit)
text_tb.on_submit(submit)
text_tc.on_submit(submit)
text_th.on_submit(submit)

plt.show()