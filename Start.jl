using Revise

# ANSI escape codes for colors and styles
const RESET = "\033[0m"
const BOLD = "\033[1m"
const UNDERLINE = "\033[4m"
const RED = "\033[31m"
const GREEN = "\033[32m"
const YELLOW = "\033[33m"
const BLUE = "\033[34m"

function simulate_3D()
    includet("Simulation3D.jl")
	@eval using .Simulation3D
	Base.invokelatest(Simulation3D.ParticleMotion)
end

function simulate_2D()
	includet("Simulation2D.jl")
	@eval using .Simulation2D
	Base.invokelatest(Simulation2D.ParticleMotion)
end

# Main loop
while true
    println("$(BOLD)$(BLUE)Choose a simulation:$(RESET)")
    println("\t$(GREEN)1: 3D Simulation$(RESET)")
    println("\t$(GREEN)2: 2D Simulation$(RESET)")
    println("$(YELLOW)Enter 'n' to exit.$(RESET)")

    choice = readline()

    if choice == "1"
        simulate_3D()
		println("$(BOLD)2D simulation is done. Now you can update the initial conditions and run it again.$(RESET)")
    elseif choice == "2"
        simulate_2D()
		println("$(BOLD)3D simulation is done. Now you can update the initial conditions and run it again.$(RESET)")
    elseif choice == "n"
        break
    else
        println("$(RED)Invalid choice. Please try again.$(RESET)")
    end
end
