import numpy as np


def B_rho(rho, z):
    return rho / (rho**2 + z**2) ** (3 / 2)


def B_z(rho, z):
    return z / (rho**2 + z**2) ** (3 / 2)


def B_phi(rho, z):
    return 0


def B_magnitude(rho, z):
    Br = B_rho(rho, z)
    Bz = B_z(rho, z)
    Bphi = B_phi(rho, z)
    return np.sqrt(Br**2 + Bphi**2 + Bz**2)


def gradient_B_magnitude(rho, z):
    B_r = rho / (rho**2 + z**2) ** (3 / 2)
    B_z = z / (rho**2 + z**2) ** (3 / 2)

    dB_r_drho = (z**2 - 2 * rho**2) / (rho**2 + z**2) ** (5 / 2)
    dB_r_dz = -3 * rho * z / (rho**2 + z**2) ** (5 / 2)

    dB_z_drho = -3 * rho * z / (rho**2 + z**2) ** (5 / 2)
    dB_z_dz = (rho**2 - 2 * z**2) / (rho**2 + z**2) ** (5 / 2)

    gradient_magnitude = np.sqrt(dB_r_drho**2 + dB_r_dz**2 + dB_z_drho**2 + dB_z_dz**2)

    return gradient_magnitude


def L_B(rho, z):
    B = B_magnitude(rho, z)
    grad_B = gradient_B_magnitude(rho, z)
    return B / grad_B


def calculate_magnetic_field(rho, z):
    B_r = rho / (rho**2 + z**2) ** (3 / 2)
    B_z = z / (rho**2 + z**2) ** (3 / 2)
    B_phi = 0
    return B_r, B_phi, B_z
