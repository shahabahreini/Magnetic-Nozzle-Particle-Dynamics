import os
import numpy as np
import plotly.graph_objects as go
import argparse
from lib import (
    search_for_export_csv,
    extract_parameters_by_file_name,
    read_exported_csv_simulation,
    read_exported_csv_simulatio_3D
)

# ---------------------------- Constants and Config ---------------------------- #
CSV_FOLDER = "/csv/"
EXPORT_IMAGE_EXTENSION = "svg"
IS_MULTI_FILES = False
ELECTRIC_FIELD_INCLUDED = True

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

def plotter(chosen_csv, save_filename, parameter_dict, plot_type):
    if plot_type == "energy":
        export_file_name = "Total_Energy"
        y_label = r"$\text{Dimensionless Total Energy}$"
        title = r"$E_{\text{total}}$ vs $\tau$"
    elif plot_type == "momentum":
        export_file_name = "Angular_Momentum"
        y_label = r"$P_\phi \text{ (Angular Momentum)}$"
        title = fr"$P_\phi$ vs $\tau$ \text{{({METHOD})}}"

    x_label = r"$\tau$"

    # Fetch CSV files
    path_ = os.path.dirname(__file__) + CSV_FOLDER if IS_MULTI_FILES else ""
    file_list = os.listdir(path_) if IS_MULTI_FILES else [chosen_csv]

    parameter_dict.update(extract_parameters_by_file_name(file_list[0]))

    fig = go.Figure()

    for fname in file_list:
        df = read_exported_csv_simulatio_3D(path_, fname)
        t_ = df["timestamp"].tolist()

        if plot_type == "energy":
            Y = (kinetic_energy_cylindricalCoordinates_electricField(df, parameter_dict["kappa"], parameter_dict["deltas"], parameter_dict["epsphi"])
                 if ELECTRIC_FIELD_INCLUDED else kinetic_energy_calculator_cylindricalCoordinates(df))
        elif plot_type == "momentum":
            Y = angular_momentum_calculator_cylindricalCoordinates(df)

        graph_label = f"$\epsilon = {parameter_dict['eps']}$"
        fig.add_trace(go.Scatter(x=t_, y=Y, mode='lines', name=graph_label))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        annotations=[
            dict(
                text=r"$\theta_0 = " + f"{parameter_dict['theta']}^\circ$ , $\alpha_0={parameter_dict['alpha']}^\circ$, "
                     r"$\beta_0 = " + f"{parameter_dict['beta']}^\circ$, $\phi_0 = 0.0^\circ$, $\kappa = {parameter_dict['kappa']}$, $\delta_* = {parameter_dict['deltas']}$, $\epsilon_\phi = {parameter_dict['epsphi']}$",
                xref="paper", yref="paper",
                x=0, y=-0.2,
                showarrow=False,
                font=dict(size=10, color="grey", style="italic")
            ),
            dict(
                text=r"$" + METHOD + r", \tau = " + f"{parameter_dict['time']}$",
                xref="paper", yref="paper",
                x=1, y=-0.2,
                showarrow=False,
                font=dict(size=10, color="grey", style="italic")
            )
        ],
        legend_title=r"$\text{Legend}$"
    )

    fig.write_image(f"plots/{save_filename}.{EXPORT_IMAGE_EXTENSION}")
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot energy or momentum from simulation data.")
    parser.add_argument("plot_type", choices=["energy", "momentum"], help="Type of plot to generate: 'energy' or 'momentum'.")
    args = parser.parse_args()

    chosen_csv = "multi_plot" if IS_MULTI_FILES else search_for_export_csv()
    plotter(chosen_csv, chosen_csv.replace(".csv", ""), PARAMETER_DICT, args.plot_type)
