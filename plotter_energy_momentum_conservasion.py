"""
Simulation Data Plotter

This script generates plots for total energy or angular momentum from simulation data.
It supports both single and multiple CSV file inputs and can handle simulations with or without electric fields.

Usage:
    python script_name.py {energy|momentum} [--use-method-legend]

Arguments:
    plot_type: Type of plot to generate ('energy' or 'momentum')
    --use-method-legend: Use method names instead of epsilon values for legend

Configuration:
    Adjust the parameters in the CONFIG section below to customize the script behavior.

Dependencies:
    - numpy
    - matplotlib
    - argparse
    - re
    - custom 'lib' module (ensure it's in the same directory or in PYTHONPATH)

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import datetime
from modules import (
    search_for_export_csv,
    extract_parameters_by_file_name,
    read_exported_csv_simulation,
    read_exported_csv_simulatio_3D,
)

# ---------------------------- CONFIG ---------------------------- #
CONFIG = {
    "CSV_FOLDER": "/high/",
    "EXPORT_IMAGE_EXTENSION": "svg",
    "IS_MULTI_FILES": False,
    "DPI": 600,
    "USE_METHOD_LEGEND": False,  # Can be overridden by command-line argument
    "DEFAULT_PARAMETERS": {
        "eps": 0.0177,
        "epsphi": 0.0,
        "kappa": 1.0,
        "deltas": 1.0,
        "beta": 90.0,
        "alpha": 30.0,
        "theta": 30.0,
        "time": 50000,
    },
    "PLOT_CONFIG": {
        "figure_dpi": 150,
        "suptitle_fontsize": 14,
        "suptitle_y": 0.98,
        "title_fontsize": 8,
        "title_color": "grey",
        "title_style": "italic",
        "electric_field_text_fontsize": 10,
        "electric_field_text_y": 0.92,
    },
    "PLOT_TYPES": {
        "energy": {
            "export_file_name": "Total_Energy",
            "y_label": r"Dimensionless Total Energy ($\tilde{E}_{total}$)",
            "suptitle": r"$\tilde{E}_{total}$ vs $\tau$",
        },
        "momentum": {
            "export_file_name": "Angular_Momentum",
            "y_label": r"Total Angular Momentum ($P_\phi$)",
            "suptitle": r"$P_\phi$ vs $\tau$",
        },
    },
    "X_LABEL": r"Dimensionless Time ($\tau$)",
}

# ---------------------------- FUNCTIONS ---------------------------- #


def kinetic_energy_calculator_cylindricalCoordinates(df_):
    vel_R, vel_phi, vel_Z = df_["dR"], df_["R"] * df_["dphi"], df_["dZ"]
    return vel_R**2 + vel_phi**2 + vel_Z**2


def kinetic_energy_cylindricalCoordinates_electricField(
    df_, kappa, delta_star, eps_phi
):
    R, Z, vel_R, vel_phi, vel_Z = (
        df_["R"],
        df_["Z"],
        df_["dR"],
        df_["R"] * df_["dphi"],
        df_["dZ"],
    )
    kinetic_energy = vel_R**2 + vel_phi**2 + vel_Z**2
    potential_energy = kappa * (
        1 - delta_star**2 * (1 - Z**2 / (R**2 + Z**2))
    ) * np.log(1 / (R**2 + Z**2)) + 0.5 * (1 - Z**2 / (R**2 + Z**2))
    return kinetic_energy + 2 * eps_phi * potential_energy


def angular_momentum_calculator_cylindricalCoordinates(df_):
    R, Z, vel_phi = df_["R"], df_["Z"], df_["R"] * df_["dphi"]
    psi = Z / np.sqrt(R**2 + Z**2)
    return vel_phi * R - psi


def format_scientific(value):
    if value:
        match = re.match(r"(\d+(?:\.\d+)?)e([+-]?\d+)", value)
        if match:
            base, exponent = match.groups()
            return f"{float(base):.1f}e{exponent}"
    return value


def extract_info_from_filename(filename):
    parts = filename.split("_")
    method = parts[0]
    reltol = next(
        (
            part.split("-", 1)[1].split(".")[0]
            for part in parts
            if part.startswith("reltol")
        ),
        None,
    )
    abstol = next(
        (
            part.split("-", 1)[1].split(".")[0]
            for part in parts
            if part.startswith("abstol")
        ),
        None,
    )
    return method, reltol, abstol


def plotter_conservation(chosen_csv, save_filename, parameter_dict, plot_type, task=None):
    """
    Enhanced plotter with progress tracking
    """
    total_steps = 6  # Total number of main steps in the process
    current_step = 0

    def update_progress(description):
        nonlocal current_step
        if task:
            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            task.description = f"[cyan]{description}"
            task.update(completed=progress_percentage)

    try:
        # Step 1: Initial Setup
        update_progress("Configuring plot parameters...")
        plt.rcParams["figure.dpi"] = CONFIG["PLOT_CONFIG"]["figure_dpi"]

        # Fetch CSV files
        path_ = (
            os.path.dirname(__file__) + CONFIG["CSV_FOLDER"]
            if CONFIG["IS_MULTI_FILES"]
            else ""
        )
        file_list = (
            [f for f in os.listdir(path_) if f.lower().endswith(".csv")]
            if CONFIG["IS_MULTI_FILES"]
            else [chosen_csv]
        )
        parameter_dict.update(extract_parameters_by_file_name(file_list[0]))

        # Step 2: Parameter Processing
        update_progress("Processing parameters...")
        electric_field_included = float(parameter_dict["epsphi"]) != 0
        plot_config = CONFIG["PLOT_TYPES"][plot_type]

        plt.suptitle(
            plot_config["suptitle"],
            fontsize=CONFIG["PLOT_CONFIG"]["suptitle_fontsize"],
            y=CONFIG["PLOT_CONFIG"]["suptitle_y"],
        )

        # Step 3: Data Processing
        update_progress("Processing data files...")
        for fname in file_list:
            df = read_exported_csv_simulatio_3D(path_, fname)
            t_ = df["timestamp"].tolist()

            if plot_type == "energy":
                Y = (
                    kinetic_energy_cylindricalCoordinates_electricField(
                        df,
                        parameter_dict["kappa"],
                        parameter_dict["deltas"],
                        parameter_dict["epsphi"],
                    )
                    if electric_field_included
                    else kinetic_energy_calculator_cylindricalCoordinates(df)
                )
            elif plot_type == "momentum":
                Y = angular_momentum_calculator_cylindricalCoordinates(df)

            method, reltol, abstol = extract_info_from_filename(fname)
            if CONFIG["USE_METHOD_LEGEND"]:
                graph_label = f"{method}"
                if reltol:
                    reltol = reltol.replace("e", "^")
                    graph_label += rf", reltol=$10^{{-{reltol.split('-')[1]}}}$"
                if abstol:
                    abstol = abstol.replace("e", "^")
                    graph_label += rf", abstol=$10^{{-{abstol.split('-')[1]}}}$"
            else:
                graph_label = rf"$\epsilon = {parameter_dict['eps']}$"

            plt.plot(t_, Y, label=graph_label)

        # Step 4: Plot Configuration
        update_progress("Configuring plot layout...")
        plt.ylabel(plot_config["y_label"])
        plt.xlabel(CONFIG["X_LABEL"])
        plt.ylim(auto=True)

        plt.title(
            rf"$\theta_0 = {parameter_dict['theta']}^{{\circ}}$ , $\alpha_0={parameter_dict['alpha']}^{{\circ}}$ , $\beta_0 = {parameter_dict['beta']}^{{\circ}}$, $\phi_0 = 0.0^{{\circ}}$, $\kappa = {parameter_dict['kappa']}$, $\delta_* = {parameter_dict['deltas']}$, $\epsilon_\phi = {parameter_dict['epsphi']}$",
            loc="right",
            fontsize=CONFIG["PLOT_CONFIG"]["title_fontsize"],
            color=CONFIG["PLOT_CONFIG"]["title_color"],
            style=CONFIG["PLOT_CONFIG"]["title_style"],
        )
        plt.title(
            f"$\\tau = {parameter_dict['time']}$",
            loc="left",
            fontsize=CONFIG["PLOT_CONFIG"]["title_fontsize"],
            color=CONFIG["PLOT_CONFIG"]["title_color"],
            style=CONFIG["PLOT_CONFIG"]["title_style"],
        )

        # Step 5: Final Touches
        update_progress("Adding final details...")
        if electric_field_included:
            plt.text(
                0.5,
                CONFIG["PLOT_CONFIG"]["electric_field_text_y"],
                "(Electric Field Included)",
                fontsize=CONFIG["PLOT_CONFIG"]["electric_field_text_fontsize"],
                ha="center",
                va="center",
                transform=plt.gcf().transFigure,
                color=CONFIG["PLOT_CONFIG"]["title_color"],
            )

        plt.legend()
        plt.tight_layout()

        # Step 6: Saving Plot
        update_progress("Saving plot...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        meaningful_name = f"{plot_type}_{parameter_dict['eps']}_{parameter_dict['kappa']}_{parameter_dict['deltas']}"
        if electric_field_included:
            meaningful_name += f"_EF_{parameter_dict['epsphi']}"

        full_filename = f"{timestamp}_{meaningful_name}_{save_filename}"
        os.makedirs("plots", exist_ok=True)
        plt.savefig(
            f"plots/{full_filename}.{CONFIG['EXPORT_IMAGE_EXTENSION']}", 
            dpi=CONFIG["DPI"]
        )
        plt.show()

        # Complete the progress
        if task:
            task.description = "[green]Plot generation completed!"
            task.update(completed=100)

        return f"plots/{full_filename}.{CONFIG['EXPORT_IMAGE_EXTENSION']}"

    except Exception as e:
        if task:
            task.description = f"[red]Error: {str(e)}"
        raise



def main():
    parser = argparse.ArgumentParser(
        description="Plot energy or momentum from simulation data."
    )
    parser.add_argument(
        "plot_type",
        choices=["energy", "momentum"],
        help="Type of plot to generate: 'energy' or 'momentum'.",
    )
    parser.add_argument(
        "--use-method-legend",
        action="store_true",
        help="Use method names instead of epsilon values for legend.",
    )
    args = parser.parse_args()

    CONFIG["USE_METHOD_LEGEND"] = args.use_method_legend

    chosen_csv = "multi_plot" if CONFIG["IS_MULTI_FILES"] else search_for_export_csv()

    # Update parameters based on the CSV file
    parameters = CONFIG["DEFAULT_PARAMETERS"].copy()
    parameters.update(extract_parameters_by_file_name(chosen_csv))

    plotter_conservation(
        chosen_csv,
        CONFIG["PLOT_TYPES"][args.plot_type]["export_file_name"],
        parameters,
        args.plot_type,
    )


if __name__ == "__main__":
    main()
