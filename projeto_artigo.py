import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constantes e Parâmetros
d_cat = 1 * 10 ** (-3)  # m
h_cat = 2 * 10 ** (-3)  # m
rho_b = 1109.0  # kg/m^3
theta = 0.0713
delta = 4.1
Cb_0 = 1  # kmol/m^3
L_total = 0.6  # m
epsilon = 0.5192
A = np.exp(28.46759622)  # m^6/kmol.kg.s
EaR = 13180.06819237  # K


def model(X, L, T, m, LHSV):
    Vp = ((np.pi * d_cat ** 2) / 4) * h_cat  # m^3
    Sp = (np.pi * d_cat ** 2) / 2 + np.pi * d_cat * h_cat  # m^2
    D_1 = (T * 2.26 * 10 ** (-10)) / 338  # m^2/s
    Deff_1 = D_1 * (theta / delta)  # m^2/s
    u1 = L_total * LHSV / (3600 * (1 - epsilon))  # m/s
    k = A * np.exp(-EaR / T)  # m^6/kmol.kg.s
    Keq = 7.97E-13 * np.exp(0.0788 * T)
    K2 = 703727.709 * np.exp(-0.0420215266 * T)  # m^3/kmol
    K4 = 10336808100 * np.exp(-0.0685249768 * T)  # m^3/kmol
    Cb_0 = 1  # kmol/m^3
    r = k * ((Cb_0 ** 2) * (1 - X) * (m - 3 * X) - (27 * X ** 4) * (Cb_0 ** 2) / (Keq * (m - 3 * X) ** 2)) / (
            1 + K2 * Cb_0 * (m - 3 * X) + K4 * Cb_0 * X)
    mod_thiele = (Vp / Sp) * np.sqrt(np.abs(r * rho_b / (Cb_0 * Deff_1)))
    n = (1 / mod_thiele) * (1 / np.tanh(3 * mod_thiele) - 1 / (3 * mod_thiele))
    dXdL = r * n * rho_b / (u1 * Cb_0)
    return dXdL


def run_simulation(parameters, L, X0):
    X_values = [odeint(model, X0, L, args=(T, m, LHSV))[:, 0] for T, m, LHSV in parameters]
    return X_values


def plot_results(L, X_values, labels, title):
    fig, ax = plt.subplots()
    for i, X in enumerate(X_values):
        ax.plot(L, X, label=labels[i])
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Conversion')
    ax.legend()
    ax.set_title(title)
    plt.show()


def main():
    X0 = 0
    L = np.linspace(0, 0.6)

    parameters_T_variation = [(347, 22.73, 0.25), (338, 22.73, 0.25), (328, 22.73, 0.25), (318, 22.73, 0.25),
                              (298, 22.73, 0.25)]
    parameters_m_variation = [(338, 22.9, 0.25), (338, 13.7, 0.25), (338, 9.16, 0.25), (338, 4.57, 0.25)]
    parameters_LHSV_variation = [(338, 22.73, 0.25), (338, 22.73, 0.76), (338, 22.73, 1.53), (338, 22.73, 2.04)]

    X_values_T_variation = run_simulation(parameters_T_variation, L, X0)
    X_values_m_variation = run_simulation(parameters_m_variation, L, X0)
    X_values_LHSV_variation = run_simulation(parameters_LHSV_variation, L, X0)

    labels_T_variation = ['T=347 K', 'T=338 K', 'T=328 K', 'T=318 K', 'T=298 K']
    labels_m_variation = ['m=22.9', 'm=13.7', 'm=9.16', 'm=4.57']
    labels_LHSV_variation = ['LHSV=0.25 h^-1', 'LHSV=0.76 h^-1', 'LHSV=1.53 h^-1', 'LHSV=2.04 h^-1']

    plot_results(L, X_values_T_variation, labels_T_variation, 'Temperature Variation')
    plot_results(L, X_values_m_variation, labels_m_variation, 'm Variation')
    plot_results(L, X_values_LHSV_variation, labels_LHSV_variation, 'LHSV Variation')


if __name__ == "__main__":
    main()
