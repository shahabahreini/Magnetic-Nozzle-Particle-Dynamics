import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import glob
import datetime
import yaml
from lib import extract_parameters_by_file_name


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
        self.share_x_axis = self.config["SHARE_X_AXIS"]
        self.calculate_traditional_magneticMoment = self.config[
            "calculate_traditional_magneticMoment"
        ]
        self.show_extremums_peaks = self.config["show_extremums_peaks"]
        self.show_amplitude_analysis = self.config["show_amplitude_analysis"]

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
share_x_axis = config.share_x_axis

# ------------------------------------ --- ----------------------------------- #

# Modern style configuration
# Modern style configuration
plt.style.use("default")  # Reset to default style
plt.rcParams.update(
    {
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "font.family": ["DejaVu Sans"],  # Use a font that supports subscripts
        "mathtext.fontset": "dejavusans",  # Ensure subscripts render correctly
    }
)


def save_plots_with_timestamp(fig, base_name, parameters=None):
    """
    Save plots with timestamp and parameters in organized directories.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    base_name : str
        Base name for the file
    parameters : dict, optional
        Dictionary of parameters to include in filename
    """
    # Generate timestamp filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file_name = f"{base_name}_{timestamp}"

    # Create plots directory if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)

    # Save with multiple extensions
    for ext in [save_file_extension, ".png"]:
        # Create subdirectory for file type if it doesn't exist
        subdir = os.path.join(plots_folder, ext.lstrip("."))
        os.makedirs(subdir, exist_ok=True)

        # Generate filename with parameters if available
        if parameters:
            param_str = "_".join([f"{k}{v}" for k, v in parameters.items()])
            filename = f"{save_file_name}_{param_str}{ext}"
        else:
            filename = f"{save_file_name}{ext}"

        path_to_save = os.path.join(subdir, filename)

        # Save figure with only supported metadata
        fig.savefig(
            path_to_save,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.1,
            metadata={
                "Creator": "Scientific Visualization Script",
                "Date": datetime.datetime.now().isoformat(),
            },
        )


def list_csv_files():
    """List all CSV files in the current directory."""
    csv_files = glob.glob("*.csv")
    print("\nAvailable CSV files:")
    for idx, file in enumerate(csv_files, 1):
        print(f"{idx}. {file}")
    return csv_files


def select_file_by_number(csv_files, prompt):
    """Select a file by entering its number."""
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(csv_files):
                return csv_files[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(csv_files)}")
        except ValueError:
            print("Please enter a valid number")


def find_first_difference(time_2d, z_2d, interpolated_z_1d, threshold):
    """Find the first point that exceeds the given threshold percentage."""
    percentage_difference = np.abs(z_2d - interpolated_z_1d) / np.abs(z_2d) * 100
    mask = percentage_difference > threshold
    if np.any(mask):
        first_exceed_index = np.where(mask)[0][0]
        return {
            "time": time_2d[first_exceed_index],
            "z_2d": z_2d[first_exceed_index],
            "z_1d": interpolated_z_1d[first_exceed_index],
            "difference": percentage_difference[first_exceed_index],
            "threshold": threshold,
        }
    return None


def create_fancy_annotation(fig, ax, xy, text, xytext):
    """Create a fancy annotation with custom styling."""
    bbox_props = dict(
        boxstyle="round,pad=0.5",
        fc="#FFFFFF",
        ec="#666666",
        alpha=0.9,
        mutation_scale=15,
    )

    arrow_props = dict(
        arrowstyle="fancy",
        color="#404040",
        connectionstyle="arc3,rad=-0.2",  # Changed from 0.2 to -0.2 for left-side arrow
        alpha=0.8,
        linewidth=1.5,
    )

    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        bbox=bbox_props,
        arrowprops=arrow_props,
        fontsize=9,
        ha="right",  # Changed from 'left' to 'right'
        va="center",
    )


# Main execution
csv_files = list_csv_files()
if len(csv_files) < 2:
    print("Error: Need at least 2 CSV files in the current directory")
    exit()

file_2d = select_file_by_number(
    csv_files, "\nSelect the 2D data file (reference, enter number): "
)
file_1d = select_file_by_number(
    csv_files, "Select the 1D data file (comparison, enter number): "
)

print(f"\nSelected files:\n2D (reference): {file_2d}\n1D (comparison): {file_1d}")

# Load and process data
df_2d = pd.read_csv(file_2d)
df_1d = pd.read_csv(file_1d)

time_2d = df_2d["timestamp"]
z_2d = df_2d["z"]
time_1d = df_1d["time"]
z_1d = df_1d["z"]

interpolated_z_1d = np.interp(time_2d, time_1d, z_1d)

# Check for differences
result = find_first_difference(time_2d, z_2d, interpolated_z_1d, 10)
if result is None:
    result = find_first_difference(time_2d, z_2d, interpolated_z_1d, 1)

# Create professional plot
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# Plot data with professional color scheme
ax.plot(
    time_2d,
    z_2d,
    label="2D Exact Z Solution (Reference)",
    color="#2E86C1",
    linewidth=2,
    alpha=0.8,
)
ax.plot(
    time_2d,
    interpolated_z_1d,
    label="1D Approximate Z Solution",
    color="#E67E22",
    linewidth=2,
    linestyle="--",
    alpha=0.8,
)

if result is not None:
    # Create fancy annotation
    annotation_text = (
        f"Error > {result['threshold']}% Difference\n"
        f"────────────────────\n"
        f"Time: {result['time']:.3f}\n"
        r"$Z_{2D}$" + f"(Exact): {result['z_2d']:.3f}\n"
        r"$Z_{1D}$" + f"(Approximated): {result['z_1d']:.3f}\n"
        f"Δ: {result['difference']:.2f}%"
    )

    create_fancy_annotation(
        fig,
        ax,
        xy=(result["time"], result["z_2d"]),
        text=annotation_text,
        xytext=(
            result["time"]
            - (time_2d.max() - time_2d.min()) * 0.1,  # Changed from + to -
            result["z_2d"] + (z_2d.max() - z_2d.min()) * 0.1,
        ),
    )

    # Add marker at the difference point
    ax.plot(
        result["time"], result["z_2d"], "o", color="#E74C3C", markersize=8, alpha=0.8
    )

    title = f"Trajectory Comparison (Error > {result['threshold']}% Difference)"
else:
    title = "Trajectory Comparison (No significant differences found)"

# Enhance plot appearance
ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
ax.set_xlabel(r"$\tau$", labelpad=10)
ax.set_ylabel(r"$\tilde z$", labelpad=10)

# Enhance legend
ax.legend(
    loc="upper right", framealpha=0.95, edgecolor="#666666", fancybox=True, shadow=True
)

# Add subtle box around the plot
for spine in ax.spines.values():
    spine.set_color("#CCCCCC")
    spine.set_linewidth(0.8)


# Define parameter mapping for latex symbols
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

# Extract parameters from the 2D file name (reference file)
parameters = extract_parameters_by_file_name(file_2d)

# Create parameter text
param_text = "\n".join(
    f"{parameter_mapping.get(key, key)}: {value}" for key, value in parameters.items()
)

# Add parameter text box
ax.text(
    0.02,
    0.95,
    "Simulation Parameters:\n" + param_text,
    transform=ax.transAxes,
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

# Adjust layout
plt.tight_layout()
save_plots_with_timestamp(fig, "1D_2D_z_solutions_comparison")
plt.show()
