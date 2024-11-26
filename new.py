import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters from the provided conditions
theta_0 = np.pi / 30  # Initial angle
alpha_0 = 90.0        # Alpha in degrees
beta_0 = 0.0
phi_0 = 0.0
epsilon = 0.005
eps_phi = 0.002
kappa = 0.01
delta_star = 0.01
time_end = 350.0
l0 = -0.9946  # Updated l0

# Derived constants
A = (kappa + 0.5) * delta_star**2
B = kappa * delta_star**2
J = epsilon

# Helper functions
def sqrt_term(z, alpha, eps_phi, A, B):
    return np.sqrt(alpha + 2 * z**2 * (A + B * np.log(np.maximum(z, 1e-10))) * eps_phi)

def denom_term(z, sqrt_val, J):
    return z**2 + (J * z**2) / (np.pi * sqrt_val)

# Differential equation for d²z/dt²
def magnetic_nozzle(t, u):
    z, dz = u
    z = max(z, 1e-10)  # Avoid division by zero
    sqrt_val = sqrt_term(z, alpha_0, eps_phi, A, B)
    denom = denom_term(z, sqrt_val, J)
    
    term1 = (np.pi * eps_phi * sqrt_val * (
        J - 2 * J * (-1 + delta_star**2) * kappa +
        2 * J * delta_star**2 * kappa * np.log(denom) +
        2 * np.pi * kappa * sqrt_val
    )) / (z * (J + np.pi * sqrt_val)**2)
    
    term2 = -(l0 + z / np.sqrt(denom)) / denom**(3/2)
    
    dzdt = term1 + term2
    return [dz, dzdt]

# Initial conditions
z0 = np.cos(theta_0)
dz0 = 0.0
t_span = (0, time_end)
u0 = [z0, dz0]

# Solve the ODE
sol = solve_ivp(magnetic_nozzle, t_span, u0, method='RK45', t_eval=np.linspace(0, time_end, 1000))

# Extract solution
t = sol.t
z = sol.y[0]

# Plotting
plt.figure(figsize=(14, 6))
plt.plot(t, z, label=r"$z(t)$", linewidth=2, c="darkcyan")
plt.title(r"$z$ vs $\tau$", fontsize=16)
plt.xlabel(r"$\tau$", fontsize=14)
plt.ylabel(r"$z$", fontsize=14)
plt.grid(True)

# Add a parameter box
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
param_text = (
    f"Simulation Parameters:\n"
    f"$\\tau$ = {time_end}\n"
    f"$\\theta_0$ = {np.degrees(theta_0):.2f}\n"
    f"$\\epsilon_\\Phi$ = {eps_phi}\n"
    f"$\\epsilon$ = {epsilon}\n"
    f"$\\beta_0$ = {beta_0}\n"
    f"$\\kappa$ = {kappa}\n"
    f"$\\delta_\\star$ = {delta_star}\n"
    f"$\\alpha_0$ = {alpha_0}\n\n"
    f"Varying Parameters:\n"
    f"Method: RK45"
)
plt.text(0.95, 0.5, param_text, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='center', horizontalalignment='right', bbox=props)

plt.legend()
plt.tight_layout()
plt.show()
