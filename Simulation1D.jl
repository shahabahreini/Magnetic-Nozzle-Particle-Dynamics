module Simulation2D

include("SharedFunctions.jl")
using .SharedFunctions
using DifferentialEquations
using Distributed
using DataFrames
using CSV

export CylindricalProblem!, SolvingtheProblem, ParticleMotion


function exportData(diffrentialSol, endTime)
    df = DataFrame(diffrentialSol)
    rename!(df, :"value1" => "dz", :"value2" => "z")

    round_endTime = round(endTime[2], digits=1)
    filename = "1D_export-eps$epsilon-epsphi$eps_phi-kappa$kappa-deltas$delta_star-beta$(round(rad2deg(beta_0)))-alpha$(round(rad2deg(alpha_0)))-theta$(round(rad2deg(theta_0)))-time$round_endTime.csv"

    CSV.write(filename, df)
    return nothing
end

# Define the second-order ODE function
function CylindricalProblem!(ddu, du, u, p, t)
    alpha, eps_phi, A, B, l0, delta_star, kappa, J, rho_0, z_0 = p
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
    exportData(sol, tspan)

    return z
end

# Main function to orchestrate the simulation
function ParticleMotion()
    # Load initial conditions
    global alpha_0, theta_0, beta_0, phi_0, epsilon, eps_phi, delta_star, kappa, rho_0, z_0, time_end, drho0, dz0, dphi0
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
