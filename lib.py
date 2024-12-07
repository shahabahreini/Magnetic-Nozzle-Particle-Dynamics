import numpy as np
from matplotlib import rcParams
from tabulate import tabulate
from colorama import Fore, Style, init
import os
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress
import matplotlib.pyplot as plt
import re
from datetime import datetime
from collections import defaultdict
from plotter_violation import load_and_calculate_variation
from plotter_velocity_components import plot_velocity_components
from scipy import integrate
from scipy.integrate import cumulative_trapezoid

console = Console()


def print_styled(text, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{text}{Style.RESET_ALL}")


def search_for_export_csv():
    # Get all files in current directory
    files = os.listdir()

    # Filter out files with csv extension
    csv_files = [file for file in files if file.endswith(".csv")]

    if not csv_files:
        print_styled("No CSV files found in the current directory.", Fore.RED)
        return None

    # Create a table of CSV files
    file_table = [[i + 1, file] for i, file in enumerate(csv_files)]

    # Print the list of csv files using tabulate
    print_styled("\nCSV files in current directory:", Fore.CYAN)
    print(tabulate(file_table, headers=["#", "Filename"], tablefmt="fancy_grid"))

    # Ask user to choose a file
    while True:
        choice = input("Choose a file (enter a number from the list): ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(csv_files):
                selected_file = csv_files[choice - 1]
                print_styled(f"Selected file: {selected_file}", Fore.GREEN)
                return selected_file
        except ValueError:
            pass
        print_styled("Invalid choice, please try again.", Fore.RED)


def extract_parameters_by_file_name(fname):
    numbers = {}

    # Adjusted regex pattern to handle scientific notation (e.g., 1.23e-8)
    pattern = r"(eps|epsphi|kappa|deltas|beta|alpha|theta|time)(\d+\.\d+(?:e[+-]?\d+)?)"

    for match in re.finditer(pattern, fname):
        key = match.group(1)
        # Converts string directly to float, handling scientific notation
        value = float(match.group(2))
        numbers[key] = value

    return numbers


def read_exported_csv_simulation(path_, fname_):
    """Gets the folder path and desired file name and load the data into Pandas DataFrame"""

    data = pd.read_csv(path_ + fname_)
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "value1",
            "value2",
            "value3",
            "value4",
            "value5",
            "value6",
        ],
    )
    df.rename(
        columns={
            "value1": "dR",
            "value2": "dphi",
            "value3": "dZ",
            "value4": "R",
            "value5": "phi",
            "value6": "Z",
        },
        inplace=True,
    )
    return df


def read_exported_csv_simulatio_3D(path_, fname_):
    """Gets the folder path and desired file name and load the data into Pandas DataFrame"""

    data = pd.read_csv(path_ + fname_)
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "drho",
            "dphi",
            "dz",
            "rho",
            "phi",
            "z",
        ],
    )
    df.rename(
        columns={
            "drho": "dR",
            "dphi": "dphi",
            "dz": "dZ",
            "rho": "R",
            "phi": "phi",
            "z": "Z",
        },
        inplace=True,
    )
    return df


def read_exported_csv_2Dsimulation(path_, fname_):
    """Gets the folder path and desired file name and load the data into Pandas DataFrame"""
    fpath = os.path.join(path_, fname_)
    data = pd.read_csv(fpath)
    df = pd.DataFrame(
        data,
        columns=["timestamp", "omega_rho", "omega_z", "rho", "z", "drho", "dz", "dphi"],
    )

    return df


def adiabtic_calculator(v_x, x, extremum_idx, label=None):
    velocity = v_x
    position = x

    # Compute the changes in the components of X
    delta_X = position.diff()

    # Compute the cumulative sum of velocity times delta_X
    adiabatic = np.cumsum(velocity * delta_X)

    # Compute the integral of V.dX between sequential extremum indexes
    integral_VdX = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        integral = np.sum(velocity[start_idx:end_idx] * delta_X[start_idx:end_idx])
        integral_VdX.append(integral)

    # Plot the integral versus cycles
    # plt.plot(range(len(integral_VdX)), integral_VdX)
    # plt.xlabel('Cycles')
    # plt.ylabel(r'$\oint\, V.\, dX$')
    # plt.title('Closed Path Integral Of Radial Velocity per Cycles')
    # plt.show()

    return integral_VdX


def adiabatic_calculator_noCycles(v_rho, rho, extremum_idx=None, label=None):
    """
    Calculate the integral of V_rho.drho using proper numerical integration

    Parameters:
    -----------
    v_rho : array_like
        Velocity component in rho direction
    rho : array_like
        Position coordinates (rho)
    extremum_idx : int, optional
        Index of extremum point if partial integration is needed
    label : str, optional
        Label for the calculation

    Returns:
    --------
    adiabatic : array_like
        Cumulative integral values
    """
    # Ensure arrays are numpy arrays
    v_rho = np.array(v_rho)
    rho = np.array(rho)

    # Input validation
    if len(v_rho) != len(rho):
        raise ValueError(
            f"Input arrays must have same length. Got v_rho: {len(v_rho)}, rho: {len(rho)}"
        )

    if extremum_idx is not None:
        if extremum_idx > len(rho):
            raise ValueError(
                f"extremum_idx ({extremum_idx}) cannot be larger than array length ({len(rho)})"
            )
        v_rho = v_rho[:extremum_idx]
        rho = rho[:extremum_idx]

    print(f"Before integration - v_rho shape: {v_rho.shape}, rho shape: {rho.shape}")

    # Calculate the integral using cumulative trapezoid method
    adiabatic = integrate.cumulative_trapezoid(v_rho, rho, initial=0)

    print(f"After integration - adiabatic shape: {adiabatic.shape}")

    # Ensure the output array has the same length as input
    if len(adiabatic) != len(rho):
        raise ValueError(
            f"Integration resulted in unexpected array length. Expected {len(rho)}, got {len(adiabatic)}"
        )

    return adiabatic


def adiabtic_calculator_fixed(v_x, x, extremum_idx, label=None):
    velocity = v_x
    position = x

    # Compute the changes in the components of X
    delta_X = position.diff()

    # Compute the cumulative sum of velocity times delta_X
    adiabatic = np.cumsum(velocity * delta_X)

    # Compute the integral of V.dX between sequential extremum indexes
    integral_VdX = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        integral = np.sum(velocity[start_idx:end_idx] * delta_X[start_idx:end_idx])
        integral_VdX.append(integral)

    # Plot the integral versus cycles
    plt.plot(range(len(integral_VdX)), integral_VdX, label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\oint\, V.\, dX$")
    plt.title("Closed Path Integral Of Radial Velocity per Cycles")

    return adiabatic


def magnetic_change_calculate(B_x, B_z, extremum_idx, label=None):
    # Calculate the magnitude of the magnetic field vector at each point
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    # Compute the relative changes in the magnitude of the magnetic field
    relative_magnetic_changes = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        initial_magnitude = B_magnitude[start_idx]
        final_magnitude = B_magnitude[end_idx - 1]
        relative_change = (
            (final_magnitude - initial_magnitude) / initial_magnitude * 100
        )
        relative_magnetic_changes.append(relative_change)

    # Plot the relative magnetic changes versus cycles
    plt.plot(
        range(len(relative_magnetic_changes)), relative_magnetic_changes, label=label
    )
    plt.axhline(
        y=-0.065, color="r", linestyle="--", label="Threshold (0.2)"
    )  # Add a dashed line at 0.2
    plt.xlabel("Cycles")
    plt.ylabel(r"Relative $\Delta B$ (%)")
    plt.title("Relative Magnetic Field Changes per Cycles")
    plt.legend()

    return relative_magnetic_changes


def epsilon_calculate(B_x, B_z, extremum_idx, time, label=None):
    # Calculate the magnitude of the magnetic field vector at each point
    """
    The epsilon_calculate function calculates the dimensionless parameter epsilon for each cycle.

    :param B_x: Calculate the magnitude of the magnetic field vector at each point
    :param B_z: Calculate the magnitude of the magnetic field vector
    :param extremum_idx: Find the indices of the local maxima and minima in b_magnitude
    :param time: Calculate the time duration of one gyration cycle
    :param label: Label the plot
    :return: A list of epsilon values for each cycle
    :doc-author: Trelent
    """
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    # Compute epsilon for each cycle
    start_idx = extremum_idx[0]
    end_idx = extremum_idx[1]
    omega_g = B_magnitude[start_idx]
    # omega_g = ((1-B_magnitude[end_idx]/B_magnitude[start_idx])**2)*B_magnitude[start_idx]# Assuming omega_g is proportional to B
    # Time duration of one gyration cycle
    tau_B = time[end_idx] - time[start_idx]
    epsilon_i = omega_g * tau_B

    epsilon_values = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        # Assuming omega_g is proportional to B
        omega_g = B_magnitude[start_idx]
        # omega_g = (B_magnitude[start_idx]**2 / (B_magnitude[end_idx])
        # Time duration of one gyration cycle
        tau_B = time[end_idx] - time[start_idx]
        epsilon = omega_g * tau_B / epsilon_i
        epsilon_values.append(epsilon)

    np.savetxt("integral.csv", np.array(epsilon_values), delimiter=",")

    # Plot epsilon versus cycles
    plt.plot(range(len(epsilon_values)), epsilon_values, label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\epsilon$")
    plt.title(r"Dimensionless Parameter $\epsilon$ per Cycles")
    plt.legend()

    return epsilon_values


def calculate_dynamic_epsilon(data, q=1, m=1, label=None):
    # Calculate velocity components
    data["v_rho"] = data["drho"]
    data["v_phi"] = data["rho"] * data["dphi"]
    data["v_z"] = data["dz"]

    # Calculate the magnitude of the magnetic field
    data["B"] = np.sqrt(data["Magnetic_rho"] ** 2 + data["Magnetic_z"] ** 2)

    # Calculate the gradient of the magnetic field using finite differences
    data["grad_B_rho"] = data["B"].diff() / data["rho"].diff()
    data["grad_B_z"] = data["B"].diff() / data["z"].diff()
    data["grad_B_rho"].fillna(0, inplace=True)  # Handle NaN values
    data["grad_B_z"].fillna(0, inplace=True)

    # Calculate the dot product of grad_B and velocity
    data["grad_B_dot_v"] = (
        data["grad_B_rho"] * data["v_rho"] + data["grad_B_z"] * data["v_z"]
    )

    # Calculate epsilon
    data["epsilon"] = (q / m) * (data["B"] ** 2) / data["grad_B_dot_v"]

    plt.plot(range(len(data["epsilon"])), data["epsilon"], label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\epsilon$")
    plt.title(r"Dimensionless Parameter $\epsilon$ per Cycles")
    plt.legend()

    return data[["timestamp", "epsilon"]]


def epsilon_calculate_allPoints(B_x, B_z, time, label=None):
    # Calculate the magnitude of the magnetic field vector at each point
    """
    The epsilon_calculate function calculates the dimensionless parameter epsilon for each cycle.

    :param B_x: Calculate the magnitude of the magnetic field vector at each point
    :param B_z: Calculate the magnitude of the magnetic field vector
    :param extremum_idx: Find the indices of the local maxima and minima in b_magnitude
    :param time: Calculate the time duration of one gyration cycle
    :param label: Label the plot
    :return: A list of epsilon values for each cycle
    :doc-author: Trelent
    """
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    # Compute epsilon for each cycle
    omega_g = B_magnitude[0]  # Assuming omega_g is proportional to B
    # Time duration of one gyration cycle
    tau_B = time[1] - time[0]
    epsilon_i = omega_g * tau_B

    epsilon_values = []
    for i in range(len(time) - 1):
        omega_g = B_magnitude[i] ** 2 / B_magnitude[i + 1]
        # Time duration of one gyration cycle
        tau_B = time[i + 1] - time[i]
        epsilon = omega_g * tau_B / epsilon_i
        epsilon_values.append(epsilon)

    np.savetxt("integral.csv", np.array(epsilon_values), delimiter=",")

    # Plot epsilon versus cycles
    plt.plot(range(len(epsilon_values)), epsilon_values, label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\epsilon$")
    plt.title(r"Dimensionless Parameter $\epsilon$ per Cycles")
    plt.legend()


def calculate_magnetic_field(rho, z):
    """
    Calculate magnetic field components in cylindrical coordinates.
    """
    B_r = rho / (rho**2 + z**2) ** (3 / 2)
    B_z = z / (rho**2 + z**2) ** (3 / 2)
    B_phi = 0
    return B_r, B_phi, B_z


def calculate_guiding_center(B, v, rho, z):
    """
    Calculate the guiding center correction.
    """
    B_mag_sq = np.dot(B, B)
    v_cross_B = np.cross(v, B)
    R_gc = v_cross_B / B_mag_sq
    r_gc_rho = rho - R_gc[0]
    r_gc_z = z - R_gc[2]
    return r_gc_rho, r_gc_z


# ------------------------------ Magnetic Field ------------------------------ #
# Define the components of the magnetic field in cylindrical coordinates
def B_rho(rho, z):
    return rho / (rho**2 + z**2) ** (3 / 2)


def B_z(rho, z):
    return z / (rho**2 + z**2) ** (3 / 2)


def B_phi(rho, z):
    return 0  # Given that B_phi = 0


# Compute the magnitude of the magnetic field
def B_magnitude(rho, z):
    Br = B_rho(rho, z)
    Bz = B_z(rho, z)
    Bphi = B_phi(rho, z)
    return np.sqrt(Br**2 + Bphi**2 + Bz**2)


# Calculate the gradient of B using numerical differentiation
def gradient_B_magnitude(rho, z):
    # Define the magnetic field components
    B_r = rho / (rho**2 + z**2) ** (3 / 2)
    B_z = z / (rho**2 + z**2) ** (3 / 2)

    # Calculate partial derivatives of B_r
    dB_r_drho = (z**2 - 2 * rho**2) / (rho**2 + z**2) ** (5 / 2)
    dB_r_dz = -3 * rho * z / (rho**2 + z**2) ** (5 / 2)

    # Calculate partial derivatives of B_z
    dB_z_drho = -3 * rho * z / (rho**2 + z**2) ** (5 / 2)
    dB_z_dz = (rho**2 - 2 * z**2) / (rho**2 + z**2) ** (5 / 2)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(dB_r_drho**2 + dB_r_dz**2 + dB_z_drho**2 + dB_z_dz**2)

    return gradient_magnitude


# Calculate the magnetic field scale length
def L_B(rho, z):
    B = B_magnitude(rho, z)
    grad_B = gradient_B_magnitude(rho, z)
    return B / grad_B


# ------------------------------------- - ------------------------------------ #


def calculate_velocity_components(B, v):
    # Calculate the magnitude (norm) of the magnetic field vector B
    norm_B = np.linalg.norm(B)

    # Ensure we avoid division by zero
    if norm_B == 0:
        raise ValueError(
            "The magnetic field magnitude is zero. Cannot calculate velocity components."
        )

    # Calculate the unit vector of the magnetic field
    unit_B = B / norm_B

    # Calculate the component of the velocity parallel to the magnetic field
    v_parallel_B = np.dot(v, unit_B) * unit_B

    # Calculate the component of the velocity perpendicular to the magnetic field
    v_perpendicular_B = v - v_parallel_B

    return v_parallel_B, v_perpendicular_B


def calculate_adiabaticity(B, v, rho, z):
    # Calculate the velocity components
    v_parallel_B, v_perpendicular_B = calculate_velocity_components(B, v)

    # Calculate the magnitude (norm) of the magnetic field vector B
    norm_B = np.linalg.norm(B)

    # Calculate the gyroradius (r_gyro = v_perpendicular / |B|)
    gyroradius = np.linalg.norm(v_perpendicular_B) / norm_B

    # Calculate the magnetic field scale length L_B (assuming L_B is defined elsewhere)
    L_B_value = L_B(rho, z)

    # Calculate the adiabaticity parameter (mu = r_gyro / L_B)
    adiabaticity = gyroradius / L_B_value

    return adiabaticity


def calculate_ad_mio(
    df,
    label=None,
    use_guiding_center=True,
    auto_scale=True,
    y_margin=1e-40,
    param_dict=None,
):
    """
    The calculate_ad_mio function calculates the adiabatic invariant mu (magnetic moment)
    for a charged particle in a magnetic field at each point in the DataFrame.
    """
    # Constants
    m = 1  # Mass of the particle (adjust as needed)

    # --------------------------- figures configuration -------------------------- #
    # Set the font to a more professional option (if available)
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

    # Increase the default font size
    rcParams["font.size"] = 10
    rcParams["axes.titlesize"] = 12
    rcParams["axes.labelsize"] = 12
    # Set a style for a more professional look
    # plt.style.use('whitegrid')
    # Set labels with enhanced styling
    # ---------------------------------------------------------------------------- #

    df["v_rho"] = df["drho"]
    df["v_phi"] = df["rho"] * df["dphi"]
    df["v_z"] = df["dz"]

    mu_values = []
    adiabaticity_values = []
    v_parallel_B_values = []
    v_perpendicular_B_values = []

    for i, row in df.iterrows():
        # Compute magnetic field components at the particle's position
        B_r, B_phi, B_z = calculate_magnetic_field(row["rho"], row["z"])
        B = np.array([B_r, B_phi, B_z])
        v = np.array([row["v_rho"], row["v_phi"], row["v_z"]])

        if use_guiding_center:
            # Compute magnetic field components at the guiding center
            r_gc_rho, r_gc_z = calculate_guiding_center(B, v, row["rho"], row["z"])
            B_r, B_phi, B_z = calculate_magnetic_field(r_gc_rho, r_gc_z)
            B = np.array([B_r, B_phi, B_z])

        # Calculate the magnitude of the magnetic field
        B_magnitude = np.linalg.norm(B)

        # Compute perpendicular velocity component
        B_unit = B / B_magnitude
        v_perp_vector = v - np.dot(v, B_unit) * B_unit
        v_perp_magnitude = np.linalg.norm(v_perp_vector)

        v_parallel_B, v_perpendicular_B = calculate_velocity_components(B, v)
        v_parallel_B_values.append(np.linalg.norm(v_parallel_B))
        v_perpendicular_B_values.append(np.linalg.norm(v_perpendicular_B))

        # adiabaticity = calculate_adiabaticity(B, v, row['rho'], row['z'])
        # mu_values.append(adiabaticity)

        # Compute mu for each point
        mu = m * v_perp_magnitude**2 / (2 * B_magnitude)
        mu_values.append(mu)

    # Save array to a CSV file
    # np.savetxt("array.csv", np.array(mu_values), delimiter=",")

    # Plotting violation of adiabatic invariant
    load_and_calculate_variation(mu_values, df["timestamp"], param_dict["eps"])

    # Plotting Velocity components
    plot_velocity_components(
        df["timestamp"],
        v_parallel_B_values,
        v_perpendicular_B_values,
        title="Sample Velocity Components",
        subtitle=None,
    )

    # Plot mu versus time points
    plt.plot(df["timestamp"], mu_values, label=label)

    # -------------------------- Improve y-axis scaling -------------------------- #
    if auto_scale:
        plt.ylim(auto=True)  # Automatically adjust y-axis limits based on data
        # Alternatively, you can set manual limits:
    else:
        plt.ylim(
            min(mu_values) - y_margin, max(mu_values) + y_margin
        )  # Add a margin if needed
    # ------------------------------------------------------------------------------- #

    print("Average mu:", np.mean(mu_values))

    # Create a more eye-catching title
    if use_guiding_center:
        title = r"Adiabatic Invariant $\mu = \frac{m v_{\perp}^2}{2 B}$ Evolution"
        subtitle = "Based on Magnetic Field at Guiding Center"
    else:
        title = r"Adiabatic Invariant $\mu = \frac{m v_{\perp}^2}{2 B}$ Evolution"
        subtitle = "Based on Magnetic Field at Particle Position"

    if param_dict["epsphi"] != 0:
        subtitle += f", $\\epsilon_{{\\phi}} = {param_dict['epsphi']}$"
    else:
        subtitle += f", no Electric Field"

    plt.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.title(subtitle, fontsize=10, fontweight="normal", style="italic")

    # Add a light gray box around the plot for emphasis
    plt.gca().patch.set_facecolor("#f0f0f0")
    plt.gcf().patch.set_facecolor("white")

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    # Optionally, add a colorbar if your plot uses colors
    # plt.colorbar(label='Value Range')
    plt.legend()

    return df["timestamp"], mu_values


def get_axis_label(param):
    labels = {
        "rho": r"$\tilde{R}$",
        "z": r"$\tilde{Z}$",
        "drho": r"$d\tilde{R}/d\tau$",
        "dz": r"$d\tilde{Z}/d\tau$",
        "timestamp": r"$\tau$",
        "omega_rho": r"$\omega_{\tilde{R}}$",
        "omega_z": r"$\omega_{\tilde{Z}}$",
        "eps": r"$\epsilon$",
        "epsphi": r"$\epsilon_{\phi}$",
        "kappa": r"$\kappa$",
        "deltas": r"$\delta_s$",
        "beta": r"$\beta$",
        "alpha": r"$\alpha$",
        "theta": r"$\theta$",
        "time": r"$\tau$",
    }
    return labels.get(param, param)


def list_folders(root="."):
    # List all directories in the root folder
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    if not folders:
        console.print("[red]No folders found in the current directory![/red]")
        exit(1)

    table = Table(title="Available Folders")
    table.add_column("#", justify="center", style="cyan", no_wrap=True)
    table.add_column("Folder", style="magenta")

    for i, folder in enumerate(folders, 1):
        table.add_row(str(i), folder)

    console.print(table)
    return folders


def list_csv_files(folder):
    # List all CSV files in the selected folder
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        console.print(f"[red]No CSV files found in the folder '{folder}'![/red]")
        exit(1)

    table = Table(title=f"\nCSV Files in '{folder}'")
    table.add_column("#", justify="center", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")

    for i, file in enumerate(files, 1):
        table.add_row(str(i), file)

    console.print(table)
    return files
