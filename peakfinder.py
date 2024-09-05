import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import os
from lib import *
import lib
import numpy as np
import datetime

# Import argreletextrema
import numpy as np
from scipy.signal import argrelextrema, find_peaks_cwt, find_peaks
from scipy.misc import electrocardiogram
from findpeaks import findpeaks
import yaml


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
            "calculate_traditional_magneticMoment"]

    def load_config(self, config_path):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)


config = Configuration('config.yaml')

# Use values from the config file
save_file_name = config.save_file_name
save_file_extension = config.save_file_extension
is_multi_files = config.is_multi_files
target_folder_multi_files = config.target_folder
plots_folder = config.plots_folder
parameter_dict = config.parameter_dict
fpath = config.target_folder
extremum_of = config.extremum_of
# ------------------------------------ --- ----------------------------------- #


def plot_extremums(results):
    df = results['df']
    plt.plot(df['x'], df['y'], label=r'$V_X$')
    plt.plot(df['x'][df['peak']], df['y'][df['peak']], 'rx', label=r'peak')
    plt.xlabel('Steps (DataPoint Index)')
    plt.ylabel(r'$V_X$')
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
        fp.plot(xlabel='Steps (DataPoint Index)', ylabel=r'$V_X$')
        fp = findpeaks(method="topology", lookahead=1)
        fp.plot(xlabel='Steps (DataPoint Index)', ylabel=r'$V_X$')
        fp.plot_persistence()

    return peak_idx


def plotter(path_, fname_):
    global parameter_dict

    # Initialize the plot before processing files
    plt.figure(figsize=(10, 6))

    plot_data = []

    if is_multi_files:
        path_ = os.path.join(os.path.dirname(__file__),
                             target_folder_multi_files)
        filelst = os.listdir(path_)
    else:
        path_ = ""
        filelst = [fname_ + ".csv"]

    for fname in filelst:
        df = lib.read_exported_csv_2Dsimulation(path_, fname)
        varibale_to_find_peaks_with = df[config.extremum_of]

        peak_idxx = peakfinder_(varibale_to_find_peaks_with)

        y_axis_data = lib.adiabtic_calculator(df["drho"], df["rho"], peak_idxx)
        x_axis_data = [df["timestamp"].tolist()[i] for i in peak_idxx[1:]]

        parameter_dict = extract_parameters_by_file_name(fname)

        eps = parameter_dict["eps"]

        # Collect data for sorting
        plot_data.append((eps, x_axis_data, y_axis_data, fname))

    # Sort the plot data by epsilon in descending order
    plot_data.sort(reverse=True, key=lambda x: x[0])

    # Plotting after sorting
    for eps, x_axis_data, y_axis_data, fname in plot_data:
        plt.plot(
            x_axis_data,
            y_axis_data,
            marker="o",
            markerfacecolor="#344e41",
            markersize=3,
            label=fr"$\epsilon = {eps}$"
        )

    plt.rcParams["figure.dpi"] = 150
    plt.ylabel(r"J=$\oint v_{x} \, \mathrm{d}x$")
    plt.xlabel(r"$\tau$")
    subtitle = "Adiabatic invariant\n" + r"J=$\oint v_{x}\,\mathrm{d}x$"
    plt.suptitle(
        subtitle,
        fontsize=12
    )
    # Move the legend to the right of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    path_to_save = os.path.join(plots_folder, str(
        save_file_name + save_file_extension))
    plt.savefig(path_to_save, dpi=600)
    path_to_save = os.path.join(plots_folder, str(save_file_name + ".png"))
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
        X, Y = lib.calculate_ad_mio(data_frame, label=label,
                                    use_guiding_center=config.based_on_guiding_center, auto_scale=auto_scale, y_margin=y_margin, param_dict=extract_parameters_by_file_name(file_path))

        return data_frame[config.extremum_of]

    # Prepare and sort file data based on 'eps' values from filenames
    file_data = [
        (extract_parameters_by_file_name(file)["eps"], f'ε={
         extract_parameters_by_file_name(file)["eps"]}', os.path.join(csv_directory, file))
        for file in os.listdir(csv_directory) if file.endswith('.csv')
    ]
    file_data.sort(key=lambda x: x[0])

    y_data = []

    # Plot adiabatic calculation results for each file
    if is_multi_files:
        for _, label, file_path in file_data:
            y_data.extend(plot_adiabatic_results(file_path, label))
    else:
        y_data = plot_adiabatic_results(
            chosen_csv, f'ε={extract_parameters_by_file_name(chosen_csv)["eps"]}')

    # Adjust plot layout to accommodate legend without overlapping plots
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.8)
    plt.show()


if not is_multi_files:
    chosen_csv = search_for_export_csv()
    indivisual_file_names_to_read = [chosen_csv]
else:
    chosen_csv = "multi_plot"

if config.calculate_integral:
    plotter(fpath, chosen_csv.replace(".csv", ""))

if config.calculate_traditional_magneticMoment:
    perform_adiabatic_calculations(chosen_csv)
