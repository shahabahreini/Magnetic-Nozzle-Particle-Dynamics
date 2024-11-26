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
	df = DataFrame(time = t_values, z = z_values, dz = dz_values)
	filename = "2D_export-time$(round(endTime, digits=1)).csv"
	CSV.write(filename, df)
	return nothing
end

# Define the second-order ODE function
function CylindricalProblem!(ddu, du, u, p, t)
	alpha, eps_phi, A, B, l0, delta_star, kappa, J, rho_0, z_0 = p
	z = u[1]

	# Ensure z is positive to avoid log(z) errors
	if z <= 0
		error("z is non-positive, cannot compute log(z).")
	end


	# ------------------------------- Calculate R^2 ------------------------------ #
	inner_sqrt = alpha + 2 * z^2 * eps_phi * (A + B * log(z))
	# Ensure inner_sqrt is positive
	if inner_sqrt <= 0
		error("inner_sqrt is not positive, cannot take square root.")
	end
	sqrt_inner = sqrt(inner_sqrt)

	R2 = (J * z^2) / (π * sqrt_inner)

	# ------------------------------ Calculate frac1 ----------------------------- #
	# frac1 = -((z / sqrt(R2 + z^2)) + l0) / (R2 + z^2)^(3 / 2)
	frac1 = ((z / sqrt(R2 + z^2)) + l0) / (R2 + z^2)^(3 / 2) * (R2 + 2 * z^2) / R2

	# ------------------------------ Calculate frac2 ----------------------------- #
	frac2 = -z * (R2 * (-2 * delta_star^2 * kappa + 2 * kappa + 1) +
				  2 * delta_star^2 * kappa * R2 * log(R2 + z^2) +
				  2 * kappa * z^2) / (R2 + z^2)^2


	# ----------------------------- Calculate ddu[1] ----------------------------- #
	ddu[1] = -frac1 - eps_phi * frac2

	return nothing
end

# Function to solve the problem
function SolvingtheProblem(CylindricalProblem, du0, u0, tspan, p)
	problem = SecondOrderODEProblem(CylindricalProblem, du0, u0, tspan, p)
	sol = solve(problem, Feagin14(), reltol = 1e-35, abstol = 1e-40)

	# Extract the solved position and velocity values
	z = [u.x[2][1] for u in sol.u]
	dz = [u.x[1][1] for u in sol.u]
	t_values = sol.t

	# Export the solution data
	exportData(t_values, z, dz, tspan[2])

	return z
end

# Main function to orchestrate the simulation
function ParticleMotion()
	# Load initial conditions
	alpha_0, theta_0, beta_0, phi_0, epsilon, eps_phi, delta_star, kappa, delta_star, rho_0, z_0, time_end, drho0, dz0, dphi0 = load_initial_conditions()

	# Calculate constants A, B, and l0
	A = (1 / 2 - kappa)
	B = 2 * kappa * delta_star^2
	psi = z_0 / sqrt(rho_0^2 + z_0^2)
	l0 = dphi0 * rho_0^2 - psi
	alpha = (3 / 4 * l0 + 1)

	# Define parameters to be passed
	p = (alpha, eps_phi, A, B, l0, delta_star, kappa, 3000, rho_0, z_0)

	# Initial conditions
	du0 = [dz0]
	u0 = [z_0]
	tspan = (0.0, time_end)

	# Solve the problem
	z = SolvingtheProblem(CylindricalProblem!, du0, u0, tspan, p)

	return nothing
end

end
