module SharedFunctions


export calculateOmega, load_initial_conditions

"""
    interpret_expression(expr::String) -> Any

    Interpret the given expression by replacing known constants and functions.
    For example, "PI" is replaced with "π", "Sin" with "sin", etc.

    Returns the evaluated result of the expression.
"""
function interpret_expression(expr)
	# Replace known constants and functions
	expr = replace(expr, "PI" => "π", "Sin" => "sin", "Cos" => "cos")

	# Evaluate the expression
	return eval(Meta.parse(expr))
end

"""
    load_initial_conditions_file(filename::String) -> Dict{String, Any}

    Load initial conditions from a file. Each line in the file should be in the format:
        variable_name = expression

    Returns a dictionary with variable names as keys and evaluated expressions as values.
"""
function load_initial_conditions_file(filename::String)	
	conditions = Dict{String, Any}()

	open(filename, "r") do file
		for line in eachline(file)
			key, expr = split(line, " = ")
			conditions[key] = interpret_expression(expr)
		end
	end

	return conditions
end

"""
    load_initial_conditions() -> Tuple

    Load initial conditions from the "initial_conditions.txt" file and set the global variables.
    The function returns a tuple containing all the loaded and computed initial conditions.
"""
function load_initial_conditions()
	# Load conditions from file
	IC = load_initial_conditions_file("initial_conditions.txt")

	# Extract conditions from the dictionary
	alpha_0 = IC["alpha_0"]
	theta_0 = IC["theta_0"]
	beta_0 = IC["beta_0"]
	phi_0 = IC["phi_0"]
	epsilon = IC["epsilon"]
	eps_phi = IC["eps_phi"]
	delta_star = IC["delta_star"]
	kappa = IC["kappa"]

	# Compute rho_0 and z_0 based on theta_0
	rho_0 = sin(IC["theta_0"])
	z_0 = cos(IC["theta_0"])

	# Convert time_end to an integer
	time_end = IC["time_end"]

	# Compute drho0, dz0, and dphi0 based on the loaded conditions
	drho0 = epsilon * (cos(alpha_0) * sin(theta_0) + sin(alpha_0) * cos(beta_0) * cos(theta_0))
	dz0 = epsilon * (cos(alpha_0) * cos(theta_0) - sin(alpha_0) * cos(beta_0) * sin(theta_0))
	dphi0 = epsilon * sin(alpha_0) * sin(beta_0) / sin(theta_0)

	return alpha_0, theta_0, beta_0, phi_0, epsilon, eps_phi, delta_star, kappa, delta_star, rho_0, z_0, time_end, drho0, dz0, dphi0
end

end
