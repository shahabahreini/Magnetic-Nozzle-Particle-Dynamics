import os
import numpy as np
import matplotlib.pyplot as plt
from lib import (
    search_for_export_csv,
    extract_parameters_by_file_name,
    read_exported_csv_simulation,
)

# ---------------------------- Constants and Config ---------------------------- #
CSV_FOLDER = "/csv/testenergy/"
EXPORT_IMAGE_EXTENSION = "svg"
IS_MULTI_FILES = False
PLOT_TYPE = "energy"  # energy or momentum
ELECTRIC_FIELD_INCLUDED = True
DPI = 600

# Default Parameters
PARAMETER_DICT = {
    "eps": 0.0177,
    "epsphi": 0.0,
    "kappa": 1.0,
    "deltas": 1.0,
    "beta": 90.0,
    "alpha": 30.0,
    "theta": 30.0,
    "time": 100,
}
METHOD = "Feagin14 Method"


def kinetic_energy_calculator_cylindricalCoordinates(df_):
    vel_R, vel_phi, vel_Z = df_["dR"], df_["R"] * df_["dphi"], df_["dZ"]
    return vel_R**2 + vel_phi**2 + vel_Z**2


def kinetic_energy_cylindricalCoordinates_electricField(df_, kappa, delta_star, eps_phi):
    R, Z, Phi, vel_R, vel_phi, vel_Z = df_["R"], df_["Z"], df_["phi"], df_["dR"], df_["R"] * df_["dphi"], df_["dZ"]
    kinetic_energy = vel_R**2 + vel_phi**2 + vel_Z**2
    potential_energy = 0.5 * (1 - Z**2 / (Z**2 + R**2)) + kappa * np.log(1 / (Z**2 + R**2)) * (1 - (1 - Z**2 / (Z**2 + R**2)) * delta_star**2)
    return kinetic_energy + 2 * eps_phi * potential_energy


def angular_momentum_calculator_cylindricalCoordinates(df_):
    R, Z, vel_phi = df_["R"], df_["Z"], df_["R"] * df_["dphi"]
    psi = Z / np.sqrt(R**2 + Z**2)
    return vel_phi * R - psi


def plotter(chosen_csv, save_filename, parameter_dict):
    # Plotter Configurations
    plt.rcParams["figure.dpi"] = 150

    if PLOT_TYPE == "energy":
        export_file_name = "Total_Energy"
        y_label = r"$Dimensionless Total Energy$"
        plt.suptitle(r"$E_{total}$ vs $\tau$", fontsize=14, y=0.98)
    elif PLOT_TYPE == "momentum":
        export_file_name = "Angular_Momentum"
        y_label = r"$P_\phi$"
        plt.suptitle(r"$P_\phi$ vs $\tau$ " + f"({METHOD})\n", fontsize=14, y=0.98)
    x_label = r"$\tau$"

    # Fetch CSV files
    path_ = os.path.dirname(__file__) + CSV_FOLDER if IS_MULTI_FILES else ""
    file_list = os.listdir(path_) if IS_MULTI_FILES else [chosen_csv]

    parameter_dict.update(extract_parameters_by_file_name(file_list[0]))

    for fname in file_list:
        df = read_exported_csv_simulation(path_, fname)
        t_ = df["timestamp"].tolist()

        if PLOT_TYPE == "energy":
            Y = (kinetic_energy_cylindricalCoordinates_electricField(df, parameter_dict["kappa"], parameter_dict["deltas"], parameter_dict["epsphi"])
                 if ELECTRIC_FIELD_INCLUDED else kinetic_energy_calculator_cylindricalCoordinates(df))
        elif PLOT_TYPE == "momentum":
            Y = angular_momentum_calculator_cylindricalCoordinates(df)

        graph_label = f"$\epsilon = {parameter_dict['eps']}$"
        plt.plot(t_, Y, label=graph_label)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(
        f"$\\theta_0 = {parameter_dict['theta']}^{{\circ}}$ , $\\alpha_0={parameter_dict['alpha']}^{{\circ}}$ , $\\beta_0 = {parameter_dict['beta']}^{{\circ}}$, $\\phi_0 = 0.0^{{\circ}}$, $\\kappa = {parameter_dict['kappa']}$, $\\delta_* = {parameter_dict['deltas']}$, $\\epsilon_\\phi = {parameter_dict['epsphi']}$",
        loc="right",
        fontsize=8,
        color="grey",
        style="italic",
    )
    plt.title(
        f"{METHOD}, $\\tau = {parameter_dict['time']}$",
        loc="left",
        fontsize=8,
        color="grey",
        style="italic",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots\{save_filename}.{EXPORT_IMAGE_EXTENSION}", dpi=DPI)
    plt.show()


if __name__ == "__main__":
    chosen_csv = "multi_plot" if IS_MULTI_FILES else search_for_export_csv()
    plotter(chosen_csv, chosen_csv.replace(".csv", ""), PARAMETER_DICT)
