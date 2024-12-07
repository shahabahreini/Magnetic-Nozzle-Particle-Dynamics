import os
import pandas as pd
import matplotlib.pyplot as plt
from lib import extract_parameters_by_file_name
import datetime
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
                "Creator": "Shahab Bahreini Jangjoo",
                "Date": datetime.datetime.now().isoformat(),
            },
        )


# Function to list all CSV files in the current directory
def list_csv_files():
    files = [f for f in os.listdir(".") if f.endswith(".csv")]
    return files


# Function to plot the data from two selected CSV files
def plot_comparison(file1, file2):
    # Load the datasets
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Extract relevant columns (rho -> R and z)
    rho_1, z_1 = data1["rho"], data1["z"]
    rho_2, z_2 = data2["rho"], data2["z"]

    # Create the comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes objects explicitly
    ax.plot(z_1, rho_1, label=f"Taken from 2D Equation", linestyle="--")
    ax.plot(z_2, rho_2, label=f"Taken from 3D Equation", linestyle="-.")

    # Add axis labels with tilde
    ax.set_xlabel(r"$\tilde{z}$", fontsize=14)
    ax.set_ylabel(r"$\tilde{R}$", fontsize=14)

    # Add title and legend
    ax.set_title("Comparison of R-z Plot for Selected Simulations", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Extract parameters from the 2D file name (reference file)
    parameters = extract_parameters_by_file_name(file1)

    # Create parameter text
    param_text = "\n".join(
        f"{parameter_mapping.get(key, key)}: {value}"
        for key, value in parameters.items()
    )

    # Add parameter text box using ax.transAxes instead of plt.transAxes
    ax.text(
        0.02,
        0.95,
        "Simulation Parameters:\n" + param_text,
        transform=ax.transAxes,  # Use ax.transAxes instead of plt.transAxes
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

    plt.tight_layout()
    save_plots_with_timestamp(
        fig, "Comprison_2D_vs_3D_"
    )  # Save the plot as an image file
    plt.show()


# Main function
if __name__ == "__main__":
    csv_files = list_csv_files()

    if len(csv_files) < 2:
        print("At least two CSV files are required for comparison.")
    else:
        print("Available CSV files:")
        for idx, file in enumerate(csv_files):
            print(f"{idx + 1}. {file}")

        try:
            # Ask the user to select two files
            choice1 = int(input("Select the number for the 2D dataset: ")) - 1
            choice2 = int(input("Select the number for the 3D dataset: ")) - 1

            if 0 <= choice1 < len(csv_files) and 0 <= choice2 < len(csv_files):
                file1 = csv_files[choice1]
                file2 = csv_files[choice2]

                # Plot the comparison
                plot_comparison(file1, file2)
            else:
                print("Invalid selection. Please run the program again.")
        except ValueError:
            print(
                "Invalid input. Please enter numbers corresponding to the file choices."
            )
