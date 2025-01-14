import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize
from modules import list_folders

# Constants
eps_phi = 0.001
kappa = 0.01
delta_star = 0.01
foldername, fname = list_folders()
# Read the CSV data
print(fname)
df = pd.read_csv(fname)

# Ask user if all plots should be shown in one figure window
show_subplots = (
    input("Show all plots in one figure window? (yes/no): ").strip().lower() == "yes"
)

save_dpi = 300


# Function to estimate l0
def estimate_l0():
    # Get initial conditions
    z0 = df["z"].iloc[0]
    v0 = df["dz"].iloc[0]

    # Calculate initial acceleration from the data
    initial_acc = np.gradient(df["dz"].values[:10], df["timestamp"].values[:10])[0]

    # Define the acceleration difference function
    def acc_diff(l0):
        z = z0
        # Full equation of motion including eps_phi term
        acc = (
            -(l0**2) / (z**3)
            + eps_phi * (2 * delta_star**2 * kappa * np.log(z**2) + 2 * kappa) / z
        )
        return (acc - initial_acc) ** 2

    # Find l0 that minimizes the difference
    l0_test = np.linspace(0.1, 1.0, 1000)
    acc_diffs = [acc_diff(l0) for l0 in l0_test]
    l0 = l0_test[np.argmin(acc_diffs)]

    return l0


# Calculate l0
l0 = estimate_l0()
print(f"Estimated l0 = {l0:.6f}")


# Calculate radial frequency from numerical data
def analyze_radial_oscillations():
    # Find peaks in rho to determine oscillation periods
    peaks, _ = find_peaks(df["rho"])

    # Calculate time differences between peaks
    peak_times = df["timestamp"][peaks]
    periods = np.diff(peak_times)
    frequencies = 2 * np.pi / periods

    return peaks, frequencies


# Get screen size for dynamic figure sizing
screen_dpi = 120  # Default DPI for screen
screen_width_inches = plt.rcParams["figure.figsize"][0]  # Default width in inches
screen_height_inches = plt.rcParams["figure.figsize"][1]  # Default height in inches

# Adjust figure size to fit the screen
fig_width = screen_width_inches
fig_height = (
    screen_height_inches * 1.5
)  # Increase height to accommodate additional plot

if show_subplots:
    # Create figure with dynamic size for subplots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, _)) = plt.subplots(
        3, 2, figsize=(fig_width, fig_height), dpi=screen_dpi
    )

    # Plot R(t) and z(t)
    ax1.plot(df["timestamp"], df["rho"], label="R(t)")
    ax1.plot(df["timestamp"], df["z"], label="z(t)")
    ax1.set_xlabel("Time (τ)")
    ax1.set_ylabel("Position")
    ax1.legend()
    ax1.set_title("Position vs Time")

    # Plot phase space R-dR
    ax2.plot(df["rho"], df["drho"])
    ax2.set_xlabel("R")
    ax2.set_ylabel("dR/dτ")
    ax2.set_title("Phase Space (R)")

    # Plot phase space z-dz
    ax3.plot(df["z"], df["dz"])
    ax3.set_xlabel("z")
    ax3.set_ylabel("dz/dτ")
    ax3.set_title("Phase Space (z)")

    # Calculate and plot ωᵣ
    peaks, frequencies = analyze_radial_oscillations()
    ax4.plot(df["timestamp"][peaks[:-1]], frequencies, "o-")
    ax4.set_xlabel("Time (τ)")
    ax4.set_ylabel("ωᵣ")
    ax4.set_title("Radial Frequency vs Time")

    # Plot dr/dt vs dz/dt
    ax5.plot(df["drho"], df["dz"])
    ax5.set_xlabel(r"$\frac{dR}{d\tau}$")
    ax5.set_ylabel(r"$\frac{dz}{d\tau}$")
    ax5.set_title("Phase Space for R and z")

    plt.tight_layout()

    # Save the figure as a high-quality image
    save_dpi = 300  # High DPI for saving
    plt.savefig("plot_output.png", dpi=save_dpi, bbox_inches="tight")
    plt.show()
else:
    # Plot R(t) and z(t) separately
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(df["timestamp"], df["rho"], label="R(t)")
    plt.plot(df["timestamp"], df["z"], label="z(t)")
    plt.xlabel("Time (τ)")
    plt.ylabel("Position")
    plt.legend()
    plt.title("Position vs Time")
    plt.savefig("position_vs_time.png", dpi=save_dpi, bbox_inches="tight")
    plt.show()

    # Plot phase space R-dR separately
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(df["rho"], df["drho"])
    plt.xlabel("R")
    plt.ylabel("dR/dτ")
    plt.title("Phase Space (R)")
    plt.savefig("phase_space_R.png", dpi=save_dpi, bbox_inches="tight")
    plt.show()

    # Plot phase space z-dz separately
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(df["z"], df["dz"])
    plt.xlabel("z")
    plt.ylabel("dz/dτ")
    plt.title("Phase Space (z)")
    plt.savefig("phase_space_z.png", dpi=save_dpi, bbox_inches="tight")
    plt.show()

    # Calculate and plot ωᵣ separately
    peaks, frequencies = analyze_radial_oscillations()
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(df["timestamp"][peaks[:-1]], frequencies, "o-")
    plt.xlabel("Time (τ)")
    plt.ylabel("ωᵣ")
    plt.title("Radial Frequency vs Time")
    plt.savefig("radial_frequency.png", dpi=save_dpi, bbox_inches="tight")
    plt.show()

    # Plot dr/dt vs dz/dt separately
    plt.figure(figsize=(fig_width, fig_height))
    plt.plot(df["drho"], df["dz"])
    plt.xlabel(r"$\frac{dR}{d\tau}$")
    plt.ylabel(r"$\frac{dz}{d\tau}$")
    plt.title("Phase Space for R and z")
    plt.savefig("phase_space_R_z.png", dpi=save_dpi, bbox_inches="tight")
    plt.show()

# Additional analysis of l0
# Plot the acceleration difference vs l0 to verify the estimate
l0_range = np.linspace(0.1, 1.0, 1000)
z0 = df["z"].iloc[0]
initial_acc = np.gradient(df["dz"].values[:10], df["timestamp"].values[:10])[0]
acc_diffs = [
    -(l0**2) / (z0**3)
    + eps_phi * (2 * delta_star**2 * kappa * np.log(z0**2) + 2 * kappa) / z0
    - initial_acc
    for l0 in l0_range
]

plt.figure(figsize=(10, 6))
plt.plot(l0_range, np.abs(acc_diffs))
plt.axvline(l0, color="r", linestyle="--", label=f"Estimated l0 = {l0:.6f}")
plt.xlabel("l0")
plt.ylabel("|Acceleration Difference|")
plt.title("l0 Estimation Verification")
plt.legend()
plt.grid(True)
plt.savefig("l0_estimation_verification.png", dpi=save_dpi, bbox_inches="tight")
plt.show()
