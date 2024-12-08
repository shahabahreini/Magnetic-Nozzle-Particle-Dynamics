import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def adiabtic_calculator(v_x, x, extremum_idx, label=None):
    velocity = v_x
    position = x
    delta_X = position.diff()
    adiabatic = np.cumsum(velocity * delta_X)

    integral_VdX = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        integral = np.sum(velocity[start_idx:end_idx] * delta_X[start_idx:end_idx])
        integral_VdX.append(integral)

    return integral_VdX


def adiabatic_calculator_noCycles(v_rho, rho, extremum_idx=None, label=None):
    v_rho = np.array(v_rho)
    rho = np.array(rho)

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
    adiabatic = integrate.cumulative_trapezoid(v_rho, rho, initial=0)
    print(f"After integration - adiabatic shape: {adiabatic.shape}")

    if len(adiabatic) != len(rho):
        raise ValueError(
            f"Integration resulted in unexpected array length. Expected {len(rho)}, got {len(adiabatic)}"
        )

    return adiabatic


def adiabtic_calculator_fixed(v_x, x, extremum_idx, label=None):
    velocity = v_x
    position = x
    delta_X = position.diff()
    adiabatic = np.cumsum(velocity * delta_X)

    integral_VdX = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        integral = np.sum(velocity[start_idx:end_idx] * delta_X[start_idx:end_idx])
        integral_VdX.append(integral)

    plt.plot(range(len(integral_VdX)), integral_VdX, label=label)
    plt.xlabel("Cycles")
    plt.ylabel(r"$\oint\, V.\, dX$")
    plt.title("Closed Path Integral Of Radial Velocity per Cycles")

    return adiabatic
