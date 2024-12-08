import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datetime
import os
from .magnetic_field import calculate_magnetic_field
from .config import Configuration

config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
config = Configuration(config_path)


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


def save_plots_with_timestamp(
    fig, base_name, parameters=None, plots_folder=config.plots_folder
):
    """
    Save plots with timestamp and parameters in specified folders.

    Args:
        fig: matplotlib figure object
        base_name: base name for the saved file
        parameters: dictionary of parameters to include in filename
        plots_folder: root folder for saving plots
    """
    # Get current timestamp
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    save_file_name = f"{base_name}_{timestamp_str}"

    # Create root plots folder
    os.makedirs(plots_folder, exist_ok=True)

    # Save in both PDF and PNG formats
    for ext in [".pdf", ".png"]:
        # Create format-specific subdirectory
        subdir = os.path.join(plots_folder, ext.lstrip("."))
        os.makedirs(subdir, exist_ok=True)

        # Construct filename with parameters if provided
        if parameters:
            param_str = "_".join([f"{k}{v}" for k, v in parameters.items()])
            filename = f"{save_file_name}_{param_str}{ext}"
        else:
            filename = f"{save_file_name}{ext}"

        path_to_save = os.path.join(subdir, filename)

        # Create metadata based on file format
        if ext == ".pdf":
            metadata = {
                "Creator": "Scientific Visualization Script",
                "CreationDate": timestamp,  # datetime object for PDF
                "Title": base_name,
                "Author": "Numerical Simulation Analysis",
                "Subject": "Particle Trajectory Analysis",
                "Keywords": "electromagnetic fields, magnetic nozzles, particle trajectory",
            }
        else:  # PNG format
            metadata = {
                "Creator": "Scientific Visualization Script",
                "CreationDate": timestamp.isoformat(),  # string for PNG
                "Title": base_name,
                "Author": "Numerical Simulation Analysis",
                "Subject": "Particle Trajectory Analysis",
                "Keywords": "electromagnetic fields, magnetic nozzles, particle trajectory",
            }

        # Save with format-specific metadata
        fig.savefig(
            path_to_save,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata=metadata,
        )


def calculate_ad_mio(
    df,
    label=None,
    use_guiding_center=True,
    auto_scale=True,
    y_margin=1e-40,
    param_dict=None,
):
    m = 1
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rcParams["font.size"] = 10
    rcParams["axes.titlesize"] = 12
    rcParams["axes.labelsize"] = 12

    df["v_rho"] = df["drho"]
    df["v_phi"] = df["rho"] * df["dphi"]
    df["v_z"] = df["dz"]

    mu_values = []
    v_parallel_B_values = []
    v_perpendicular_B_values = []

    for i, row in df.iterrows():
        B_r, B_phi, B_z = calculate_magnetic_field(row["rho"], row["z"])
        B = np.array([B_r, B_phi, B_z])
        v = np.array([row["v_rho"], row["v_phi"], row["v_z"]])

        if use_guiding_center:
            r_gc_rho, r_gc_z = calculate_guiding_center(B, v, row["rho"], row["z"])
            B_r, B_phi, B_z = calculate_magnetic_field(r_gc_rho, r_gc_z)
            B = np.array([B_r, B_phi, B_z])

        B_magnitude = np.linalg.norm(B)
        B_unit = B / B_magnitude
        v_perp_vector = v - np.dot(v, B_unit) * B_unit
        v_perp_magnitude = np.linalg.norm(v_perp_vector)

        v_parallel_B, v_perpendicular_B = calculate_velocity_components(B, v)
        v_parallel_B_values.append(np.linalg.norm(v_parallel_B))
        v_perpendicular_B_values.append(np.linalg.norm(v_perpendicular_B))

        mu = m * v_perp_magnitude**2 / (2 * B_magnitude)
        mu_values.append(mu)

    plt.plot(df["timestamp"], mu_values, label=label)

    if auto_scale:
        plt.ylim(auto=True)
    else:
        plt.ylim(min(mu_values) - y_margin, max(mu_values) + y_margin)

    print("Average mu:", np.mean(mu_values))

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
    plt.gca().patch.set_facecolor("#f0f0f0")
    plt.gcf().patch.set_facecolor("white")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.legend()

    return df["timestamp"], mu_values
