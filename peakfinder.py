import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import os
from lib import *
import lib
import numpy as np
import datetime

# Import necessary libraries
from scipy.signal import argrelextrema
from findpeaks import findpeaks
import yaml
from plotter_violation import load_and_calculate_variation
import matplotlib.patches as mpatches


# ---------------------------------- Config ---------------------------------- #
class Configuration:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.save_file_name = self.config["save_file_name"]
        self.save_file_extension = self.config["save_file_extension"]
        self.is_multi_files = self.config["is_multi_files"]
        self.target_folder = self.config["target_folder_multi_files"]
        self.plots_folder = self.config["plots_folder"]
        self.parameter_dict = self.config["simulation_parameters"]
        self.extremum_of = self.config["extremum_of"]
        self.based_on_guiding_center = self.config["based_on_guiding_center"]
        self.calculate_integral = self.config["calculate_integral"]
        self.calculate_traditional_magneticMoment = self.config[
            "calculate_traditional_magneticMoment"
        ]
        self.show_extremums_peaks = self.config["show_extremums_peaks"]

    def load_config(self, config_path):
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)


config = Configuration("config.yaml")

# Use values from the config file
save_file_name = config.save_file_name
save_file_extension = config.save_file_extension
is_multi_files = config.is_multi_files
target_folder_multi_files = config.target_folder
plots_folder = config.plots_folder
parameter_dict = config.parameter_dict
fpath = config.target_folder
extremum_of = config.extremum_of
show_extremums_peaks = config.show_extremums_peaks
# ------------------------------------ --- ----------------------------------- #


def plot_extremums(results):
    df = results["df"]
    plt.plot(df["x"], df["y"], label=r"$V_X$")
    plt.plot(df["x"][df["peak"]], df["y"][df["peak"]], "rx", label=r"peak")
    plt.xlabel("Steps (DataPoint Index)")
    plt.ylabel(r"$V_X$")
    plt.legend()
    plt.show()


def spherical_to_cartesian(rho, theta, phi):
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return x, y, z


def peakfinder_(X, show_plot=True):
    # Initialize
    def peak_couter(results):
        df_result = results["df"]

        count_valleys = df_result[df_result["valley"] == True].shape[0]
        count_peaks = df_result[df_result["peak"] == True].shape[0]

        chosen_key = "peak" if count_peaks < count_valleys else "valley"
        peak_indexes = df_result[df_result[chosen_key] == True]["x"]
        peak_indexes = peak_indexes.tolist()

        return peak_indexes

    fp = findpeaks(method="peakdetect", lookahead=1)
    # fp = findpeaks(method="topology", lookahead=50)
    results = fp.fit(X)
    peak_idx = peak_couter(results)

    # Plot
    if show_plot:
        plot_extremums(results)
        fp.plot(xlabel="Steps (DataPoint Index)", ylabel=r"$V_X$")
        fp = findpeaks(method="topology", lookahead=1)
        fp.plot(xlabel="Steps (DataPoint Index)", ylabel=r"$V_X$")
        fp.plot_persistence()

    return peak_idx


def angular_momentum_calculator_cylindricalCoordinates(df_):
    R, z, vel_phi = df_["rho"], df_["z"], df_["rho"] * df_["dphi"]
    psi = z / np.sqrt(R**2 + z**2)
    return vel_phi * R - psi


def omega_squared(z, l0):
    return 4 * (3 / 4 * l0 + 1) / z**4


def adiabatic_condition(z, dz_dt, l0):
    omega = np.sqrt(omega_squared(z, l0))
    return np.abs(dz_dt / z) / (omega / 2 * np.pi)


def calculate_adiabatic_condition(df):
    z = df["z"].values
    t = df["timestamp"].values
    dz_dt = np.gradient(z, t)
    l0 = angular_momentum_calculator_cylindricalCoordinates(df)
    return adiabatic_condition(z, dz_dt, l0), dz_dt


def calculate_adiabatic_condition_electric(df, epsilon_phi, K=5):
    print(epsilon_phi)
    # Extract z and timestamp from the dataframe
    z = df["z"].values
    t = df["timestamp"].values

    # Calculate dz/dt using numerical differentiation
    dz_dt = np.gradient(z, t)

    # Calculate the generalized angular momentum l0
    l0 = angular_momentum_calculator_cylindricalCoordinates(df)

    # Calculate A and B constants
    A = (0.75 * l0) + 1.0
    B = epsilon_phi * K

    # Calculate omega and its derivative
    omega = np.sqrt(2) * np.sqrt(A + B * z**2) / z**2
    domega_dt = np.gradient(omega, t)

    # Calculate the adiabatic condition
    # Compute the term inside the absolute value
    term1 = (B * z) / (A + B * z**2)
    term2 = -2.0 / z
    adiabatic_condition_value = np.abs((term1 + term2) * dz_dt)

    return adiabatic_condition_value, dz_dt


def plotter(path_, fname_, show_growth_rate=False):
    global parameter_dict

    plot_data = []
    adiabatic_data = []

    if is_multi_files:
        path_ = os.path.join(os.path.dirname(__file__), target_folder_multi_files)
        filelst = os.listdir(path_)
    else:
        path_ = ""
        filelst = [fname_ + ".csv"]

    for fname in filelst:
        # File handling remains the same
        if is_multi_files:
            file_path = os.path.join(fpath, fname)
        else:
            file_path = os.path.join(path_, fname)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = lib.read_exported_csv_2Dsimulation(path_, fname)
        varibale_to_find_peaks_with = df[config.extremum_of]

        peak_idxx = peakfinder_(varibale_to_find_peaks_with, show_extremums_peaks)

        y_axis_data = lib.adiabtic_calculator(df["drho"], df["rho"], peak_idxx)
        x_axis_data = [df["timestamp"].tolist()[i] for i in peak_idxx[1:]]

        parameter_dict = extract_parameters_by_file_name(fname)
        eps = parameter_dict.get("eps", "N/A")

        adiabatic_cond, dz_dt = calculate_adiabatic_condition(df)
        growth_rate = np.gradient(adiabatic_cond, df["timestamp"].values)

        plot_data.append((eps, x_axis_data, y_axis_data, fname))
        adiabatic_data.append(
            (eps, df["timestamp"].values, adiabatic_cond, growth_rate)
        )

    # Sort data for plotting
    plot_data.sort(reverse=True, key=lambda x: x[0])
    adiabatic_data.sort(reverse=True, key=lambda x: x[0])

    # Create two subplots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # Plot 1: Adiabatic invariant
    for eps, x_axis_data, y_axis_data, fname in plot_data:
        ax1.plot(
            x_axis_data,
            y_axis_data,
            marker="o",
            markersize=5,
            label=rf"$\epsilon = {eps}$",
        )

    # Retrieve parameters from the first file
    if plot_data:
        first_fname = plot_data[0][3]
        parameters = extract_parameters_by_file_name(first_fname)

        # Mapping parameter keys to LaTeX representations
        parameter_mapping = {
            "eps": r"$\epsilon$",
            "epsphi": r"$\epsilon_\phi$",
            "kappa": r"$\kappa$",
            "deltas": r"$\delta_s$",
            "beta": r"$\beta_0$",
            "alpha": r"$\alpha_0$",
            "theta": r"$\theta_0$",
            "time": r"$\tau$",
        }

        # Format parameters into a string with Greek symbols
        param_text = "\n".join(
            f"{parameter_mapping.get(key, key)}: {value}"
            for key, value in parameters.items()
        )

        param_text = "Simulation Parameters:\n" + param_text

        # Add text box with parameters
        ax1.text(
            0.02,
            0.95,
            param_text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax1.set_ylabel(r"$J = \oint v_{x} \, \mathrm{d}x$", fontsize=14)
    ax1.set_title("Adiabatic Invariant", fontsize=16)
    ax1.legend(loc="upper right", fontsize=12)
    ax1.grid(True)

    # Plot 2: Adiabatic condition and growth rate
    for eps, t, adiabatic_cond, growth_rate in adiabatic_data:
        ax2.plot(t, adiabatic_cond, label=rf"$\eta$, $\epsilon = {eps}$")
        if show_growth_rate:
            ax2.plot(
                t,
                growth_rate,
                linestyle="--",
                label=rf"Growth Rate, $\epsilon = {eps}$",
            )

    # Add shaded tolerance regions with increased transparency
    regions = [
        (0, 0.001, "Highly Adiabatic (<0.001)", "green"),
        (0.001, 0.01, "Moderately Adiabatic (0.001-0.01)", "yellow"),
        (0.01, 0.1, "Approaching Breakdown (0.01-0.1)", "orange"),
        (0.1, 1, "Non-Adiabatic (>0.1)", "red"),
    ]

    # Create custom legend entries for shaded regions
    custom_patches = []
    for ymin, ymax, label, color in regions:
        ax2.axhspan(ymin, ymax, facecolor=color, alpha=0.1)
        patch = mpatches.Patch(color=color, alpha=0.3, label=label)
        custom_patches.append(patch)

    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\tau$", fontsize=14)
    ax2.set_ylabel(r"$\eta$", fontsize=14)
    ax2.set_title(
        "Adiabatic Condition" + (" and Growth Rate" if show_growth_rate else ""),
        fontsize=16,
    )
    # Combine existing legend handles with custom patches
    handles, labels = ax2.get_legend_handles_labels()
    handles.extend(custom_patches)
    labels.extend([patch.get_label() for patch in custom_patches])
    ax2.legend(handles, labels, loc="lower right", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    save_file_name = "Adiabatic_Condition_and_Growth_Rate"
    path_to_save = os.path.join(plots_folder, save_file_name + save_file_extension)
    plt.savefig(path_to_save, dpi=600)
    path_to_save = os.path.join(plots_folder, save_file_name + ".png")
    plt.savefig(path_to_save, dpi=600)
    plt.show()


def perform_adiabatic_calculations(chosen_csv, auto_scale=True, y_margin=1e-17):
    """
    Performs adiabatic calculations on datasets from CSV files located in a specific directory,
    and plots the results. Assumes the presence of specific columns in the CSV for calculations.

    Parameters:
    - chosen_csv (str): Path to the chosen CSV file or directory containing multiple CSV files.
    - auto_scale (bool): If True, auto-scale y-axis. If False, set limits based on data.
    - y_margin (float): Margin to add to y-axis limits when auto_scale is False (as a fraction of the range).
    """
    csv_directory = os.path.join(os.path.dirname(__file__), "csv")
    X = []
    Y = []

    def plot_adiabatic_results(file_path: str, label: str):
        global X, Y
        """
        Plots adiabatic calculation results for a given dataset.

        Parameters:
        - file_path (str): Path to the dataset (CSV file).
        - label (str): Label for the plot derived from the dataset.
        """
        data_frame = pd.read_csv(file_path)

        peak_finder = findpeaks(method="peakdetect", lookahead=1)
        results = peak_finder.fit(data_frame[config.extremum_of])

        # Assuming calculate_ad_mio is a comprehensive function handling all necessary plotting
        X, Y = lib.calculate_ad_mio(
            data_frame,
            label=label,
            use_guiding_center=config.based_on_guiding_center,
            auto_scale=auto_scale,
            y_margin=y_margin,
            param_dict=extract_parameters_by_file_name(file_path),
        )

        return data_frame[config.extremum_of]

    # Prepare and sort file data based on 'eps' values from filenames
    file_data = [
        (
            extract_parameters_by_file_name(file)["eps"],
            f'ε = {
         extract_parameters_by_file_name(file)["eps"]}',
            os.path.join(csv_directory, file),
        )
        for file in os.listdir(csv_directory)
        if file.endswith(".csv")
    ]
    file_data.sort(key=lambda x: x[0])

    y_data = []

    # Plot adiabatic calculation results for each file
    if is_multi_files:
        for _, label, file_path in file_data:
            y_data.extend(plot_adiabatic_results(file_path, label))
    else:
        y_data = plot_adiabatic_results(
            chosen_csv, f'ε={extract_parameters_by_file_name(chosen_csv)["eps"]}'
        )

    # Adjust plot layout to accommodate legend without overlapping plots
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.8)
    plt.show()


if __name__ == "__main__":
    if not is_multi_files:
        chosen_csv = search_for_export_csv()
        chosen_csv = os.path.basename(chosen_csv).replace(".csv", "")
    else:
        chosen_csv = "multi_plot"

    if config.calculate_integral:
        plotter(fpath, chosen_csv)

    if config.calculate_traditional_magneticMoment:
        perform_adiabatic_calculations(chosen_csv)
