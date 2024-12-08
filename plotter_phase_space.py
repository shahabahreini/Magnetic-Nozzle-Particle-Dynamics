import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize

# Constants
eps_phi = 0.001
kappa = 0.01
delta_star = 0.01

# Read the CSV data
df = pd.read_csv(
    "3D_export-eps0.0002-epsphi0.0-kappa0.01-deltas0.01-beta0.0-alpha90.0-theta3.0-time3000.0.csv"
)


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


# Plot trajectories and phase space
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

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

plt.tight_layout()
plt.show()


# Calculate adiabatic invariant J
def calculate_J():
    # Calculate J = ∮ p_R dR ≈ area of phase space orbit
    E_R = 0.5 * (df["drho"] ** 2) + 0.5 * (df["omega_rho"] ** 2 * df["rho"] ** 2)
    J = 2 * np.pi * E_R / df["omega_rho"]
    return J


J = calculate_J()

# Plot J vs time
plt.figure(figsize=(10, 6))
plt.plot(df["timestamp"], J)
plt.xlabel("Time (τ)")
plt.ylabel("J (Adiabatic Invariant)")
plt.title("Adiabatic Invariant vs Time")
plt.show()

# Print statistical analysis
print(f"Average ωᵣ = {np.mean(frequencies):.6f} ± {np.std(frequencies):.6f}")
print(f"Average J = {np.mean(J):.6f} ± {np.std(J):.6f}")

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
plt.show()
