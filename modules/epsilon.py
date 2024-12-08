import numpy as np
import matplotlib.pyplot as plt


def epsilon_calculate(B_x, B_z, extremum_idx, time, label=None):
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    start_idx = extremum_idx[0]
    end_idx = extremum_idx[1]
    omega_g = B_magnitude[start_idx]
    tau_B = time[end_idx] - time[start_idx]
    epsilon_i = omega_g * tau_B

    epsilon_values = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        omega_g = B_magnitude[start_idx]
        tau_B = time[end_idx] - time[start_idx]
        epsilon = omega_g * tau_B / epsilon_i
        epsilon_values.append(epsilon)

    np.savetxt("integral.csv", np.array(epsilon_values), delimiter=",")

    plt.plot(range(len(epsilon_values)), epsilon_values, label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\epsilon$")
    plt.title(r"Dimensionless Parameter $\epsilon$ per Cycles")
    plt.legend()

    return epsilon_values


def epsilon_calculate_allPoints(B_x, B_z, time, label=None):
    B_magnitude = np.sqrt(B_x**2 + B_z**2)
    omega_g = B_magnitude[0]
    tau_B = time[1] - time[0]
    epsilon_i = omega_g * tau_B

    epsilon_values = []
    for i in range(len(time) - 1):
        omega_g = B_magnitude[i] ** 2 / B_magnitude[i + 1]
        tau_B = time[i + 1] - time[i]
        epsilon = omega_g * tau_B / epsilon_i
        epsilon_values.append(epsilon)

    np.savetxt("integral.csv", np.array(epsilon_values), delimiter=",")

    plt.plot(range(len(epsilon_values)), epsilon_values, label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\epsilon$")
    plt.title(r"Dimensionless Parameter $\epsilon$ per Cycles")
    plt.legend()


def calculate_dynamic_epsilon(data, q=1, m=1, label=None):
    data["v_rho"] = data["drho"]
    data["v_phi"] = data["rho"] * data["dphi"]
    data["v_z"] = data["dz"]

    data["B"] = np.sqrt(data["Magnetic_rho"] ** 2 + data["Magnetic_z"] ** 2)

    data["grad_B_rho"] = data["B"].diff() / data["rho"].diff()
    data["grad_B_z"] = data["B"].diff() / data["z"].diff()
    data["grad_B_rho"].fillna(0, inplace=True)
    data["grad_B_z"].fillna(0, inplace=True)

    data["grad_B_dot_v"] = (
        data["grad_B_rho"] * data["v_rho"] + data["grad_B_z"] * data["v_z"]
    )

    data["epsilon"] = (q / m) * (data["B"] ** 2) / data["grad_B_dot_v"]

    plt.plot(range(len(data["epsilon"])), data["epsilon"], label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\epsilon$")
    plt.title(r"Dimensionless Parameter $\epsilon$ per Cycles")
    plt.legend()

    return data[["timestamp", "epsilon"]]
