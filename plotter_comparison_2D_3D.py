import os
import pandas as pd
import matplotlib.pyplot as plt
from modules import extract_parameters_by_file_name, list_csv_files
import datetime
import yaml
from modules import save_plots_with_timestamp, list_csv_files_noFolder


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
    csv_files = list_csv_files_noFolder()

    if len(csv_files) < 2:
        print("At least two CSV files are required for comparison.")
    else:
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
