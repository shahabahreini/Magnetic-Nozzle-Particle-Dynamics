import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import datetime

# Import necessary libraries
from scipy.signal import argrelextrema
from findpeaks import findpeaks
import yaml
from plotter_violation import load_and_calculate_variation
import matplotlib.patches as mpatches
from scipy.integrate import cumulative_trapezoid
from modules import (
    save_plots_with_timestamp,
    Configuration,
    calculate_ad_mio,
    read_exported_csv_2Dsimulation,
    adiabtic_calculator,
    extract_parameters_by_file_name,
    search_for_export_csv,
)

config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = Configuration(config_path)


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


def peakfinder_(X, show_plot=False):
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


# def adiabatic_condition(z, dz_dt, l0):
#     omega = np.sqrt(omega_squared(z, l0))
#     return np.abs(dz_dt / z) / (omega / 2 * np.pi)


# def calculate_adiabatic_condition(df):
#     z = df["z"].values
#     t = df["timestamp"].values
#     dz_dt = np.gradient(z, t)
#     l0 = angular_momentum_calculator_cylindricalCoordinates(df)
#     return adiabatic_condition(z, dz_dt, l0), dz_dt


def calculate_omega0_squared(l_0):
    """
    Calculate ω₀² according to the formula: ω₀ = (3/4 * l₀ + 1)
    Returns the square of ω₀
    """
    omega_0 = 3 / 4 * l_0 + 1
    return omega_0**2


def omega_tau(z, eps_phi, kappa, delta_star, l_0):
    """
    Calculate ω(τ) according to the new formula
    """
    omega0_sq = calculate_omega0_squared(l_0)

    # First term: ω₀²/z⁴
    magnetic_term = omega0_sq / (z**4)

    # Electric terms inside ε_φ parentheses
    electric_terms = (
        (kappa * delta_star**2) / (z**4)  # k δ_*²/z⁴
        - kappa * np.log(1 / (z**2))  # -k log(1/z²)
        + 1 / (z**4)  # 1/z⁴
    )

    return np.sqrt(magnetic_term + eps_phi * electric_terms)


def domega_dtau(z, z_prime, eps_phi, kappa, delta_star, l_0):
    """
    Calculate dω/dτ according to the new formula
    """
    omega = omega_tau(z, eps_phi, kappa, delta_star, l_0)

    term1 = (
        -4
        * (calculate_omega0_squared(l_0) + eps_phi * (kappa * delta_star**2 + 1))
        / (z**5)
    )
    term2 = 2 * eps_phi * kappa / z

    return (term1 + term2) * z_prime / (2 * omega)


def adiabatic_condition(z, dz_dt, eps_phi, kappa, delta_star, l_0):
    """
    Calculate the adiabatic parameter η using the new formula:
    η = (1/ω²)|dω/dt|
    """
    # Calculate ω
    omega = omega_tau(z, eps_phi, kappa, delta_star, l_0)
    # Calculate dω/dτ
    domega = domega_dtau(z, dz_dt, eps_phi, kappa, delta_star, l_0)

    # Calculate η
    eta = np.abs(domega) / (omega**2)

    return eta


def calculate_adiabatic_condition(df, fname):
    """
    Calculate adiabatic condition from DataFrame data.
    """
    # Extract parameters from filename
    params = extract_parameters_by_file_name(fname)

    # Map parameters from filename
    eps_phi = params.get("epsphi", params.get("eps"))
    kappa = params.get("kappa")
    delta_star = params.get("deltas")
    l0 = angular_momentum_calculator_cylindricalCoordinates(df)

    # Extract position and velocity data
    z = df["z"].values
    dz_dt = df["dz"].values

    # Calculate adiabatic condition
    eta = adiabatic_condition(z, dz_dt, eps_phi, kappa, delta_star, l0)
    print("The firdt calculated η: ", eta[0])

    return eta, dz_dt


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
    """
    Enhanced plotter function for visualizing adiabatic conditions and related data.

    Parameters:
        path_ (str): Base path for data files
        fname_ (str): Base filename for single file mode
        show_growth_rate (bool): Whether to show growth rate in the second subplot
    """
    # Reset to default style and then customize
    plt.style.use("default")

    # Set custom style parameters
    plt.rcParams.update(
        {
            # Font settings
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            # Grid settings
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.color": "#CCCCCC",
            # Axis settings
            "axes.linewidth": 0.5,
            "axes.edgecolor": "#333333",
            # Figure settings
            "figure.dpi": 150,
            "savefig.dpi": 600,
            # Legend settings
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "legend.fancybox": True,
            # Additional professional settings
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "figure.autolayout": True,
            "axes.axisbelow": True,
        }
    )

    plot_data = []
    adiabatic_data = []
    file_data = []

    # File handling
    if config.is_multi_files:
        path_ = os.path.join(os.path.dirname(__file__), config.target_folder)
        filelst = os.listdir(path_)
    else:
        path_ = ""
        filelst = [fname_ + ".csv"]

    # Data collection
    for fname in filelst:
        if config.is_multi_files:
            file_path = os.path.join(fpath, fname)
        else:
            file_path = os.path.join(path_, fname)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        file_data.append(file_path)

        # Read and process data
        df = read_exported_csv_2Dsimulation(path_, fname)
        varibale_to_find_peaks_with = df[config.extremum_of]
        peak_idxx = peakfinder_(
            varibale_to_find_peaks_with, config.show_extremums_peaks
        )

        # Calculate necessary values
        y_axis_data = adiabtic_calculator(df["drho"], df["rho"], peak_idxx)
        print("The first calculated J: ", y_axis_data[0])
        x_axis_data = [df["timestamp"].tolist()[i] for i in peak_idxx[1:]]

        parameter_dict = extract_parameters_by_file_name(fname)
        eps = parameter_dict.get("eps", "N/A")

        adiabatic_cond, dz_dt = calculate_adiabatic_condition(df, fname)
        growth_rate = np.gradient(adiabatic_cond, df["timestamp"].values)

        plot_data.append((eps, x_axis_data, y_axis_data, fname))
        adiabatic_data.append(
            (eps, df["timestamp"].values, adiabatic_cond, growth_rate)
        )

    # Sort data for consistent plotting
    plot_data.sort(reverse=True, key=lambda x: x[0])
    adiabatic_data.sort(reverse=True, key=lambda x: x[0])

    # Create figure with enhanced layout
    fig = plt.figure(figsize=(12, 10), dpi=150)

    if config.share_x_axis:
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax1.tick_params(labelbottom=False)
        plt.setp(ax1.get_xticklabels(), visible=False)
    else:
        gs = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

    # Generate professional color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))

    # Plot 1: Adiabatic invariant
    for (eps, x_axis_data, y_axis_data, fname), color in zip(plot_data, colors):
        ax1.plot(
            x_axis_data,
            y_axis_data,
            marker="o",
            markersize=4,
            color=color,
            label=rf"$\epsilon = {eps}$",
            linewidth=1.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
        )

    # Enhanced grid styling for both plots
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle="--", alpha=0.7, which="major", color="#CCCCCC")
        ax.grid(True, linestyle=":", alpha=0.4, which="minor", color="#EEEEEE")
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#333333")

    # Parameter text box
    if plot_data:
        first_fname = plot_data[0][3]
        parameters = extract_parameters_by_file_name(first_fname)

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

        param_text = "\n".join(
            f"{parameter_mapping.get(key, key)}: {value}"
            for key, value in parameters.items()
        )

        ax1.text(
            0.02,
            0.95,
            "Simulation Parameters:\n" + param_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.9,
                linewidth=0.5,
            ),
        )

    # First plot styling
    ax1.set_ylabel(r"$J = \oint v_{R} \, \mathrm{d}R$", fontsize=12)
    ax1.set_title("Adiabatic Invariant", pad=15)
    ax1.legend(loc="upper right", framealpha=0.9, edgecolor="#CCCCCC", fancybox=True)

    # Define adiabatic regions
    regions = [
        (0, 0.001, "Highly Adiabatic (<0.001)", "green"),
        (0.001, 0.01, "Moderately Adiabatic (0.001-0.01)", "yellow"),
        (0.01, 0.1, "Approaching Breakdown (0.01-0.1)", "orange"),
        (0.1, 1, "Non-Adiabatic (>0.1)", "red"),
    ]

    # Define adiabatic regions
    regions = [
        (0, 0.001, "Highly Adiabatic (<0.001)", "green"),
        (0.001, 0.01, "Moderately Adiabatic (0.001-0.01)", "yellow"),
        (0.01, 0.1, "Approaching Breakdown (0.01-0.1)", "orange"),
        (0.1, 1, "Non-Adiabatic (>0.1)", "red"),
    ]

    # Plot adiabatic regions
    custom_patches = []
    for ymin, ymax, label, color in regions:
        ax2.axhspan(ymin, ymax, facecolor=color, alpha=0.1, edgecolor="none")
        patch = mpatches.Patch(color=color, alpha=0.3, label=label, linewidth=0)
        custom_patches.append(patch)

    # Plot adiabatic conditions with crossing point annotations
    for (eps, t, adiabatic_cond, growth_rate), color in zip(adiabatic_data, colors):
        # Plot the main line
        ax2.plot(
            t,
            adiabatic_cond,
            color=color,
            linewidth=1.5,
            label=rf"$\eta$, $\epsilon = {eps}$",
        )

        # Find first crossing points for each region boundary
        region_boundaries = [0.001, 0.01, 0.1]
        annotated_boundaries = set()  # Keep track of which boundaries we've annotated

        for i in range(len(t) - 1):
            for boundary in region_boundaries:
                if boundary in annotated_boundaries:
                    continue  # Skip if we've already annotated this boundary

                if (adiabatic_cond[i] <= boundary <= adiabatic_cond[i + 1]) or (
                    adiabatic_cond[i + 1] <= boundary <= adiabatic_cond[i]
                ):
                    # Linear interpolation to find exact crossing point
                    frac = (boundary - adiabatic_cond[i]) / (
                        adiabatic_cond[i + 1] - adiabatic_cond[i]
                    )
                    cross_time = t[i] + frac * (t[i + 1] - t[i])

                    # Add annotation with longer arrow and horizontal text
                    ax2.annotate(
                        f"τ={cross_time:.2f}\nη={boundary:.3f}",
                        xy=(cross_time, boundary),
                        xytext=(15, 45),  # Horizontal offset
                        textcoords="offset points",
                        ha="left",
                        va="center",
                        bbox=dict(
                            boxstyle="round,pad=0.5",
                            fc="white",
                            ec="gray",
                            alpha=0.8,
                            linewidth=0.5,
                        ),
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="arc3,rad=0.2",
                            color="gray",
                            alpha=0.6,
                            lw=1.5,  # Longer arrow line width
                            shrinkA=5,  # Length of the arrow
                        ),
                    )

                    # Mark this boundary as annotated
                    annotated_boundaries.add(boundary)

                    # If we've found all boundaries, we can break the inner loop
                    if len(annotated_boundaries) == len(region_boundaries):
                        break

            # If we've found all boundaries, we can break the outer loop
            if len(annotated_boundaries) == len(region_boundaries):
                break

        if show_growth_rate:
            ax2.plot(
                t,
                growth_rate,
                color=color,
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                label=rf"Growth Rate, $\epsilon = {eps}$",
            )

    # Second plot styling
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$\eta$", fontsize=12)
    ax2.set_title(
        "Adiabatic Condition" + (" and Growth Rate" if show_growth_rate else ""), pad=5
    )

    # X-axis labels based on sharing setting
    if config.share_x_axis:
        ax2.set_xlabel(r"$\tau$", fontsize=12)
    else:
        ax1.set_xlabel(r"$\tau$", fontsize=12)
        ax2.set_xlabel(r"$\tau$", fontsize=12)

    # Combine legends
    handles, labels = ax2.get_legend_handles_labels()
    handles.extend(custom_patches)
    labels.extend([patch.get_label() for patch in custom_patches])
    ax2.legend(
        handles,
        labels,
        loc="lower right",
        framealpha=0.9,
        edgecolor="#CCCCCC",
        fancybox=True,
        ncol=1,
    )

    # Final adjustments
    plt.tight_layout()

    # Get parameters from first file if available
    parameters = None
    if file_data:
        parameters = extract_parameters_by_file_name(os.path.basename(file_data[0]))

    # Save plots with timestamp
    save_plots_with_timestamp(fig, "Adiabatic_Condition_and_Growth_Rate", parameters)

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
        X, Y = calculate_ad_mio(
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
            f'ε = {extract_parameters_by_file_name(file)["eps"]}',
            os.path.join(csv_directory, file),
        )
        for file in os.listdir(csv_directory)
        if file.endswith(".csv")
    ]
    file_data.sort(key=lambda x: x[0])

    y_data = []

    # Plot adiabatic calculation results for each file
    if config.is_multi_files:
        for _, label, file_path in file_data:
            y_data.extend(plot_adiabatic_results(file_path, label))
    else:
        y_data = plot_adiabatic_results(
            chosen_csv, f'ε={extract_parameters_by_file_name(chosen_csv)["eps"]}'
        )

    # Adjust plot layout to accommodate legend without overlapping plots
    plt.legend(loc="lower left", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.8)
    # plt.show()


def calculate_amplitude_and_average(df, peak_indices):
    """
    Calculate amplitude per period and running average.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the time series data
    peak_indices : list
        List of indices corresponding to peaks

    Returns:
    --------
    timestamps : list
        List of timestamps for each period
    amplitudes : list
        List of amplitudes for each period
    running_averages : list
        List of running averages
    """
    timestamps = []
    amplitudes = []
    running_averages = []

    for i in range(1, len(peak_indices)):
        start_idx = peak_indices[i - 1]
        end_idx = peak_indices[i]

        # Calculate amplitude for this period
        period_data = df[config.extremum_of].iloc[start_idx:end_idx]
        amplitude = period_data.max() - period_data.min()

        # Use the midpoint of the period for the timestamp
        timestamp = df["timestamp"].iloc[start_idx + (end_idx - start_idx) // 2]

        timestamps.append(timestamp)
        amplitudes.append(amplitude)

        # Calculate running average
        running_avg = np.mean(amplitudes)
        running_averages.append(running_avg)

    return timestamps, amplitudes, running_averages


def plot_amplitude_analysis(ax, timestamps, amplitudes, running_averages, color):
    """
    Plot amplitude analysis on the given axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to plot on
    timestamps : list
        List of timestamps for x-axis
    amplitudes : list
        List of amplitudes per period
    running_averages : list
        List of running averages
    color : str/tuple
        Color for the plots
    """
    # Plot amplitude per period
    ax.plot(
        timestamps,
        amplitudes,
        marker="o",
        linestyle="-",
        color=color,
        alpha=0.6,
        label=r"Amplitude per period",
    )

    # Plot running average
    ax.plot(
        timestamps,
        running_averages,
        linestyle="--",
        color=color,
        linewidth=2,
        label=r"Running average",
    )

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Amplitude")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()


def plot_amplitude_analysis_separate(path_, fname_, show_plot=False):
    """
    Create a separate plot for amplitude analysis with dual y-axes and enhanced visuals.
    Legend positioned inside the graph area in the upper left corner.
    """
    # Set up the style
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 9,  # Smaller font for compact legend
            "figure.titlesize": 18,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.color": "#CCCCCC",
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#333333",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "figure.figsize": (10, 6),
        }
    )

    # Create figure and axes
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # File handling
    if config.is_multi_files:
        path_ = os.path.join(os.path.dirname(__file__), target_folder_multi_files)
        filelst = os.listdir(path_)
    else:
        path_ = ""
        filelst = [fname_ + ".csv"]

    # Generate color palette
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(filelst)))

    # Lists to store legend handles and labels
    lines1, lines2 = [], []
    labels1, labels2 = [], []

    for fname, color in zip(filelst, colors):
        if config.is_multi_files:
            file_path = os.path.join(fpath, fname)
        else:
            file_path = os.path.join(path_, fname)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Extract parameters from filename
        current_params = extract_parameters_by_file_name(fname)
        eps_value = current_params.get("eps", "N/A")

        # Read and process data
        df = read_exported_csv_2Dsimulation(path_, fname)
        varibale_to_find_peaks_with = df[config.extremum_of]
        peak_indices = peakfinder_(varibale_to_find_peaks_with, False)

        # Calculate amplitudes
        timestamps, amplitudes, running_averages = calculate_amplitude_and_average(
            df, peak_indices
        )

        # Plot amplitude per period on left y-axis
        line1 = ax1.plot(
            timestamps,
            amplitudes,
            marker="o",
            markersize=4,
            linestyle="-",
            linewidth=1.5,
            color=color,
            alpha=0.7,
            markerfacecolor="white",
            markeredgewidth=1,
            markeredgecolor=color,
            label=rf"Amp. ($\epsilon={eps_value}$)",
        )
        lines1.extend(line1)
        labels1.append(rf"Amp. ($\epsilon={eps_value}$)")

        # Plot running average on right y-axis
        line2 = ax2.plot(
            timestamps,
            running_averages,
            linestyle="--",
            linewidth=2,
            color=color,
            alpha=0.9,
            label=rf"Avg. ($\epsilon={eps_value}$)",
        )
        lines2.extend(line2)
        labels2.append(rf"Avg. ($\epsilon={eps_value}$)")

    # Enhance axes labels and title
    ax1.set_xlabel(r"$\tau$", fontsize=14, labelpad=10)
    ax1.set_ylabel(r"Amplitude per period", fontsize=14, labelpad=10)
    ax2.set_ylabel(r"Running average", fontsize=14, labelpad=15)

    # Adjust grid
    ax1.grid(True, linestyle="--", alpha=0.4, which="major")
    ax1.grid(True, linestyle=":", alpha=0.2, which="minor")
    ax1.minorticks_on()

    # Title with enhanced styling
    plt.title("Amplitude Analysis", pad=20, fontweight="bold")

    # Combine all lines and labels for the legend
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    # Create legend inside the plot area
    # Note: Using ax1 instead of fig for the legend
    legend = ax1.legend(
        all_lines,
        all_labels,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.95),  # Adjusted to be inside the plot
        ncol=1,
        fancybox=True,
        shadow=True,
        framealpha=0.8,
        edgecolor="gray",
        facecolor="white",
        borderpad=0.5,
        labelspacing=0.3,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # Add a box around the legend
    legend.get_frame().set_linewidth(0.5)

    # Set equal aspect ratio for y-axes
    y1_min, y1_max = ax1.get_ylim()
    y2_min, y2_max = ax2.get_ylim()
    ax1_range = y1_max - y1_min
    ax2_range = y2_max - y2_min

    # Adjust y-axis limits to make them proportional
    ax1.set_ylim(y1_min - 0.1 * ax1_range, y1_max + 0.1 * ax1_range)
    ax2.set_ylim(y2_min - 0.1 * ax2_range, y2_max + 0.1 * ax2_range)

    # Add light spines
    for spine in ax1.spines.values():
        spine.set_linewidth(0.5)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()

    # Get parameters from first file if available
    save_parameters = None
    if filelst:
        save_parameters = extract_parameters_by_file_name(filelst[0])

    # Save plots with timestamp
    save_plots_with_timestamp(fig, "Amplitude_Analysis", save_parameters)

    if show_plot:
        plt.show()


if __name__ == "__main__":
    if not config.is_multi_files:
        chosen_csv = search_for_export_csv()
        chosen_csv = os.path.basename(chosen_csv).replace(".csv", "")
    else:
        chosen_csv = "multi_plot"

    if config.calculate_integral:
        plotter(config.target_folder, chosen_csv)

    if config.calculate_traditional_magneticMoment:
        perform_adiabatic_calculations(chosen_csv)

    if config.show_amplitude_analysis:
        plot_amplitude_analysis_separate(config.target_folder, chosen_csv)
