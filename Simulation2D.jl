module Simulation2D

# Import necessary modules and functions
include("SharedFunctions.jl")
using .SharedFunctions
using DifferentialEquations
using ArbNumerics
using Distributed
using DataFrames
using CSV

export CylindricalProblem!, SolvingtheProblem, ParticleMotion


"""
    exportData(diffrentialSol, endTime)

	Export the solution data to a CSV file.
	- Renames columns for clarity.
	- Calculates omega values for rho and z.
	- Constructs a filename based on various parameters.
	- Writes the DataFrame to a CSV file.
"""
function exportData(diffrentialSol, endTime)
    df = DataFrame(diffrentialSol)
    rename!(df, :"value1" => "drho", :"value2" => "dz", :"value3" => "rho", :"value4" => "z")

    round_endTime = round(endTime[2], digits=1)
    filename = "2D_export-eps$epsilon-epsphi$eps_phi-kappa$kappa-deltas$delta_star-beta$(round(rad2deg(beta_0)))-alpha$(round(rad2deg(alpha_0)))-theta$(round(rad2deg(theta_0)))-time$round_endTime.csv"

    CSV.write(filename, df)
    return nothing
end



"""
    CylindricalProblem!(ddu, du, u, p, t)

	Calculate the differential equations for the cylindrical problem in 2D.
	- Uses global variables for initial conditions.
	- Computes the differential equations based on the current state.
"""
function CylindricalProblem!(ddu, du, u, p, t)
    global z_0, rho_0, dphi0
    rho, z = u

    # ---------------------------- Electric Field Term --------------------------- #
    log_term = log(rho^2 + z^2)

    # Corrected dPhi_dz
    dPhi_dz = -z * (rho^2 * (-2 * delta_star^2 * kappa + 2 * kappa + 1) +
                    2 * delta_star^2 * kappa * rho^2 * log_term +
                    2 * kappa * z^2) / (rho^2 + z^2)^2

    # Corrected dPhi_dR
    dPhi_dR = rho * (2 * (delta_star^2 - 1) * kappa * rho^2 +
                     2 * delta_star^2 * kappa * z^2 * log_term +
                     (1 - 2 * kappa) * z^2) / (rho^2 + z^2)^2

    # ------------------------------ Exact Equations ----------------------------- #
    # Calculate the l0, frac0, and frac1 terms for efficiency
    l0 = epsilon * sin(alpha_0) * sin(beta_0) * sin(theta_0) - z_0 / sqrt(z_0^2 + rho_0^2)
    fac0 = sqrt(z^2 + rho^2)
    fac1 = (rho^2 + z^2)^(3 / 2)
    fac3 = (l0 + z / fac0)

    # Update the differential equations
    ddu[1] = 1 / rho^3 * fac3 * (l0 + z * (2 * rho^2 + z^2) / fac1) - eps_phi * dPhi_dR
    ddu[2] = -1 / fac1 * fac3 - eps_phi * dPhi_dz

    return nothing
end

"""
    SolvingtheProblem(CylindricalProblem, du0, u0, tspan)

	Solve the differential equations for the given initial conditions.
	- Uses the CylindricalProblem! function to compute the differential equations.
	- Extracts the solved position values.
	- Exports the solution data to a CSV file.
"""
function SolvingtheProblem(CylindricalProblem, du0, u0, tspan)
    problem = SecondOrderODEProblem{true}(CylindricalProblem, du0, u0, tspan, SciMLBase.NullParameters())
    sol = solve(problem, Feagin14(), reltol=1e-35, abstol=1e-40)

    # Extract the solved position values
    rho = sol[3, :]
    z = sol[4, :]

    # Export the solution data
    exportData(sol, tspan)

    return rho, z
end

"""
    ParticleMotion()

	Simulate the motion of a particle based on the initial conditions.
	- Loads the initial conditions.
	- Sets up the initial state and time span.
	- Solves the differential equations using the SolvingtheProblem function.
"""
function ParticleMotion()
    global alpha_0, theta_0, beta_0, phi_0, epsilon, eps_phi, delta_star, kappa, delta_star, rho_0, z_0, time_end, drho0, dz0, dphi0
    alpha_0, theta_0, beta_0, phi_0, epsilon, eps_phi, delta_star, kappa, delta_star, rho_0, z_0, time_end, drho0, dz0, dphi0 = load_initial_conditions()

    du0 = [drho0; dz0]
    u0 = [rho_0; z_0]
    tspan = (0.0, time_end)
    rho, z = SolvingtheProblem(CylindricalProblem!, du0, u0, tspan)

    return nothing
end

end