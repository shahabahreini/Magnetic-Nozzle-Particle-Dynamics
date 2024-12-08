import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from .magnetic_field import calculate_magnetic_field, L_B


def calculate_velocity_components(B, v):
    norm_B = np.linalg.norm(B)

    if norm_B == 0:
        raise ValueError(
            "The magnetic field magnitude is zero. Cannot calculate velocity components."
        )

    unit_B = B / norm_B
    v_parallel_B = np.dot(v, unit_B) * unit_B
    v_perpendicular_B = v - v_parallel_B

    return v_parallel_B, v_perpendicular_B


def calculate_guiding_center(B, v, rho, z):
    B_mag_sq = np.dot(B, B)
    v_cross_B = np.cross(v, B)
    R_gc = v_cross_B / B_mag_sq
    r_gc_rho = rho - R_gc[0]
    r_gc_z = z - R_gc[2]
    return r_gc_rho, r_gc_z


def calculate_adiabaticity(B, v, rho, z):
    v_parallel_B, v_perpendicular_B = calculate_velocity_components(B, v)
    norm_B = np.linalg.norm(B)
    gyroradius = np.linalg.norm(v_perpendicular_B) / norm_B
    L_B_value = L_B(rho, z)
    adiabaticity = gyroradius / L_B_value
    return adiabaticity


def magnetic_change_calculate(B_x, B_z, extremum_idx, label=None):
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

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

    plt.plot(
        range(len(relative_magnetic_changes)), relative_magnetic_changes, label=label
    )
    plt.axhline(y=-0.065, color="r", linestyle="--", label="Threshold (0.2)")
    plt.xlabel("Cycles")
    plt.ylabel(r"Relative $\Delta B$ (%)")
    plt.title("Relative Magnetic Field Changes per Cycles")
    plt.legend()

    return relative_magnetic_changes
