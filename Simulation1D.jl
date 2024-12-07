module Simulation2D

include("SharedFunctions.jl")
using .SharedFunctions
using DifferentialEquations
using Distributed
using DataFrames
using CSV

export CylindricalProblem!, SolvingtheProblem, ParticleMotion

# Function to export data
function exportData(t_values, z_values, dz_values, endTime)
    df = DataFrame(time=t_values, z=z_values, dz=dz_values)
    filename = "2D_export-time$(round(endTime, digits=1)).csv"
    CSV.write(filename, df)
    return nothing
end

# Define the second-order ODE function
function CylindricalProblem!(ddu, du, u, p, t)
    alpha, eps_phi, A, B, l0, delta_star, kappa, J, rho_0, z_0 = p

    # ------------------------------ First Approach ------------------------------ #
    # C = 2 * delta_star^2 * kappa * log(u[1]^2) - 2 * kappa + 1
    # ddu[1] = -J * sqrt(eps_phi) / (π * u[1]^2) * (2 * kappa * delta_star^2 - C) / sqrt(C)

    # ddu[1] = 2 * J / π * 4 * (3 / 4 * l0 + 1) / u[1]^3
    # ddu[1] = 1 / u[1]^3 * (2 * J * 4 * (3 / 4 * l0 + 1) + eps_phi * (2 * kappa - 2 * kappa * delta_star^2 - 1))

    # ------------------------------ Second Approach ----------------------------- #
    # omega_0 = 2 * sqrt(3 / 4 * l0 + 1)

    # Fixed equation: replaced delta_start with delta_star and corrected parentheses
    # ddu[1] = -J / π * (eps_phi * (-2 * delta_star^2 * kappa + kappa * u[1]^4 - 2) - 2 * omega_0^2) /
    #          (u[1]^3 * sqrt(omega_0^2 + eps_phi * (delta_star^2 * kappa - kappa * u[1]^4 * log(1 / u[1]^2) + 1))) -
    #          0.0 * (2 * kappa - 2 * kappa * delta_star^2 - 1)
    # ------------------------------ Third Approach ------------------------------ #
    # R² term replacement: J/(π*omega)
    # omega_0 = 2 * sqrt(3 / 4 * l0 + 1)
    # omega = sqrt(omega_0^2 / u[1]^4 + eps_phi * (kappa * delta_star^2 / u[1]^4 - kappa * log(1 / u[1]^2) + 1 / u[1]^4))
    # R_squared = J / (π * omega)

    # # Numerator terms
    # term1 = 2.0 * (delta_star^2 - 1.0) * kappa * R_squared
    # term2 = 2.0 * delta_star^2 * kappa * u[1]^2 * log(R_squared + u[1]^2)
    # term3 = (1.0 - 2.0 * kappa) * u[1]^2

    # # Denominator
    # denominator = (R_squared + u[1]^2)^2

    # # Final equation
    # ddu[1] = -J / π * (eps_phi * (-2 * delta_star^2 * kappa + kappa * u[1]^4 - 2) - 2 * omega_0^2) /
    #          (u[1]^3 * sqrt(omega_0^2 + eps_phi * (delta_star^2 * kappa - kappa * u[1]^4 * log(1 / u[1]^2) + 1))) +
    #          eps_phi * (term1 + term2 + term3) / denominator
    # ------------------------------ Fourth Approach ----------------------------- #
    # R² term replacement: J/(π*omega)
    # omega_0 = 2 * sqrt(3 / 4 * l0 + 1)
    # omega = sqrt(omega_0^2 / u[1]^4 + eps_phi * (kappa * delta_star^2 / u[1]^4 - kappa * log(1 / u[1]^2) + 1 / u[1]^4))
    # R_squared = J / (π * omega)

    # # Numerator terms
    # term1 = 2.0 * (delta_star^2 - 1.0) * kappa * R_squared
    # term2 = 2.0 * delta_star^2 * kappa * u[1]^2 * log(R_squared + u[1]^2)
    # term3 = (1.0 - 2.0 * kappa) * u[1]^2
    # fac0 = sqrt(u[1]^2 + R_squared)
    # fac1 = (R_squared + u[1]^2)^(3 / 2)
    # fac3 = (l0 + u[1] / fac0)

    # # Denominator
    # denominator = (R_squared + u[1]^2)^2
    # ddu[1] = 1 / fac1 * fac3 - eps_phi * (term1 + term2 + term3) / denominator
    # ------------------------------ Fifth Approach ------------------------------ #
    omega_0 = 2 * sqrt(3 / 4 * l0 + 1)
    omega = sqrt(omega_0^2 / u[1]^4 + eps_phi * (kappa * delta_star^2 / u[1]^4 - kappa * log(1 / u[1]^2) + 1 / u[1]^4))
    ddu[1] = (2 * J / π) * (omega_0^2 + eps_phi * (kappa * delta_star^2 + 1 - π / (2 * J) * kappa * u[1]^2)) / (u[1]^5 * omega) + 2 * kappa * eps_phi / u[1]


    return nothing
end

# Function to solve the problem
function SolvingtheProblem(CylindricalProblem, du0, u0, tspan, p)
    problem = SecondOrderODEProblem(CylindricalProblem, du0, u0, tspan, p)
    sol = solve(problem, Feagin14(), reltol=1e-35, abstol=1e-40)

    # Extract the solved position and velocity values
    z = [u[2] for u in sol.u]  # Simplified array comprehension
    dz = [u[1] for u in sol.u]  # Simplified array comprehension
    t_values = sol.t

    # Export the solution data
    exportData(t_values, z, dz, tspan[2])

    return z
end

# Main function to orchestrate the simulation
function ParticleMotion()
    # Load initial conditions
    alpha_0, theta_0, beta_0, phi_0, epsilon, eps_phi, delta_star, kappa, delta_star, rho_0, z_0, time_end, drho0, dz0, dphi0 = load_initial_conditions()  # Removed duplicate delta_star

    # Calculate constants A, B, and l0
    A = (1 / 2 - kappa)
    B = 2 * kappa * delta_star^2
    psi = z_0 / sqrt(rho_0^2 + z_0^2)
    l0 = dphi0 * rho_0^2 - psi
    alpha = (3 / 4 * l0 + 1)

    # Ask user for J value
    print("Enter J value (default=0.000012542): ")
    J = tryparse(Float64, readline())
    J = isnothing(J) ? 0.000012542 : J  # Use default if invalid input

    # Define parameters to be passed
    p = (alpha, eps_phi, A, B, l0, delta_star, kappa, J, rho_0, z_0)

    # Initial conditions
    du0 = [dz0]
    u0 = [z_0]
    tspan = (0.0, time_end)

    # Solve the problem
    z = SolvingtheProblem(CylindricalProblem!, du0, u0, tspan, p)

    return nothing
end

end
