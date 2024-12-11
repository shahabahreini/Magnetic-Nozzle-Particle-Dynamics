using Revise
using Printf
using REPL
using Pkg

# Terminal control functions
function clear_screen()
    if Sys.iswindows()
        run(`cmd /c cls`)
    else
        print("\033[2J\033[H")
        flush(stdout)
    end
end

function clear_line()
    if Sys.iswindows()
        print("\r")
    else
        print("\033[2K\r")
    end
    flush(stdout)
end

# Terminal styling constants
const STYLES = Dict(
    :reset => "\033[0m",
    :bold => "\033[1m",
    :dim => "\033[2m",
    :italic => "\033[3m",
    :underline => "\033[4m",
    :blink => "\033[5m",
    :reverse => "\033[7m",
    :hidden => "\033[8m",
    :strike => "\033[9m"
)

const COLORS = Dict(
    :black => "\033[30m",
    :red => "\033[31m",
    :green => "\033[32m",
    :yellow => "\033[33m",
    :blue => "\033[34m",
    :magenta => "\033[35m",
    :cyan => "\033[36m",
    :white => "\033[37m"
)

const BG_COLORS = Dict(
    :black => "\033[40m",
    :red => "\033[41m",
    :green => "\033[42m",
    :yellow => "\033[43m",
    :blue => "\033[44m",
    :magenta => "\033[45m",
    :cyan => "\033[46m",
    :white => "\033[47m"
)

# Utility functions for styling
style(text, styles...) = join([STYLES[s] for s in styles], "") * text * STYLES[:reset]
color(text, col) = COLORS[col] * text * STYLES[:reset]
bgcolor(text, col) = BG_COLORS[col] * text * STYLES[:reset]

# Simulation status structure
mutable struct SimulationStatus
    dimension::Int
    progress::Float64
    status::String
    start_time::Float64
end

# Modified Progress indicator
function show_progress(message::String, progress::Float64)
    bar_width = 30
    filled = round(Int, progress * bar_width)
    bar = "█"^filled * "░"^(bar_width - filled)
    percentage = round(Int, progress * 100)
    clear_line()
    print("\r$message [$bar] $percentage%")
    flush(stdout)
end

# Modified Simulation functions
function run_simulation(dim::Int)
    status = SimulationStatus(dim, 0.0, "Initializing...", time())

    # Load appropriate module
    module_name = "Simulation$(dim)D"
    includet("$module_name.jl")
    @eval using .$(Symbol(module_name))

    println(color("\n▶ Starting $(dim)D Simulation...", :blue))
    println(style("━"^50, :dim))

    # Actual simulation call
    try
        Base.invokelatest(getfield(eval(Symbol(module_name)), :ParticleMotion))
    catch e
        println(color("\n⚠ Simulation Error: $(sprint(showerror, e))", :red))
        return
    end

    execution_time = round(time() - status.start_time, digits=2)
    println("\n" * style("━"^50, :dim))
    println(color("✓ $(dim)D simulation completed in $execution_time seconds", :green))
    println(color("  You can now update initial conditions and run again.", :cyan))
end

# Modified Menu rendering
function render_menu_header()
    clear_screen()
    title = " Particle Motion Simulator "
    border_width = 40
    padding = div(border_width - length(title), 2)

    println(style("\n╔" * "═"^border_width * "╗", :bold))
    println(style("║", :bold) * " "^padding * color(title, :cyan) * " "^(border_width - padding - length(title)) * style("║", :bold))
    println(style("╚" * "═"^border_width * "╝", :bold))
end

function render_menu_options()
    options = [
        (key="1", label="1D Simulation", desc="Linear particle motion"),
        (key="2", label="2D Simulation", desc="Planar particle motion"),
        (key="3", label="3D Simulation", desc="Spatial particle motion"),
        (key="p", label="Package Management", desc="Install/update required packages"),
        (key="h", label="Help", desc="Show documentation"),
        (key="q", label="Quit", desc="Exit the simulator")
    ]

    println()  # Add spacing
    for opt in options
        key_display = color("[$(opt.key)]", :yellow)
        label_display = " $(opt.label)"
        desc_display = style("\n      $(opt.desc)", :dim)
        println("  $key_display$label_display$desc_display")
    end
end

# Modified Help function
function show_help()
    clear_screen()
    border_width = 50

    println(style("\n📚 Help Documentation", :bold))
    println(style("\n" * "━"^border_width, :dim))

    sections = [
        ("Simulation Modes", [
            "• 1D Simulation: Simulates particle motion along a single axis",
            "• 2D Simulation: Simulates particle motion in a plane",
            "• 3D Simulation: Simulates particle motion in three-dimensional space"
        ]),
        ("Controls", [
            "• Use number keys (1-3) to select simulation mode",
            "• Press 'h' for this help screen",
            "• Press 'q' to quit the application"
        ])
    ]

    for (title, items) in sections
        println(color("\n$title:", :cyan))
        for item in items
            println(item)
        end
    end

    println(style("\n" * "━"^border_width, :dim))
    print("\nPress Enter to return to main menu...")
    readline()
end

# Package management functions
function check_and_install_packages()
    clear_screen()
    println(color("\n📦 Package Management", :cyan))
    println(style("━"^50, :dim))

    # Read dependencies from install_packages.jl
    deps_file = joinpath(dirname(@__FILE__), "install_packages.jl")
    if !isfile(deps_file)
        println(color("\n⚠ install_packages.jl not found!", :red))
        return
    end

    # Parse dependencies from file
    deps_content = read(deps_file, String)
    deps_match = match(r"dependencies\s*=\s*\[(.*?)\]"s, deps_content)
    if deps_match === nothing
        println(color("\n⚠ Could not parse dependencies from install_packages.jl!", :red))
        return
    end

    # Extract package names
    packages = map(m -> strip(replace(m.match, "\"" => "")),
        eachmatch(r"\"(.*?)\"", deps_match.captures[1]))

    # Check installed packages using the new installed() function
    installed_pkgs = installed()
    installed_status = Dict{String,Bool}()
    missing_packages = String[]

    println(color("\nChecking installed packages...", :blue))
    for pkg in packages
        pkg_name = split(pkg, " ")[1]  # Handle cases with version specifications
        is_installed = haskey(installed_pkgs, pkg_name)
        installed_status[pkg_name] = is_installed
        if !is_installed
            push!(missing_packages, pkg_name)
        end
    end

    # Display status
    println("\nPackage Status:")
    println(style("━"^30, :dim))

    for (pkg, is_installed) in installed_status
        status = is_installed ?
                 color("✓ Installed", :green) :
                 color("○ Not installed", :yellow)
        println("$(rpad(pkg, 20)) $status")
    end

    # Handle missing packages
    if !isempty(missing_packages)
        println("\n" * color("Missing packages found:", :yellow))
        for pkg in missing_packages
            println("  • $pkg")
        end

        print("\nWould you like to install missing packages? (y/N): ")
        if lowercase(strip(readline())) == "y"
            println(color("\nInstalling missing packages...", :blue))
            for pkg in missing_packages
                try
                    print("Installing $pkg... ")
                    Pkg.add(pkg)
                    println(color("✓", :green))
                catch e
                    println(color("⚠ Failed!", :red))
                    println(color("  Error: $(sprint(showerror, e))", :red))
                end
            end
        end
    else
        println(color("\n✓ All packages are installed!", :green))
    end

    println("\nPress Enter to return to main menu...")
    readline()
end

# Package utility functions
function dependencies()
    deps = Pkg.dependencies()
    return deps isa Dict ? deps : Dict(deps)
end

function installed()
    deps = dependencies()
    installs = Dict{String,VersionNumber}()
    for (uuid, dep) in deps
        dep.is_direct_dep || continue
        dep.version === nothing && continue
        installs[dep.name] = dep.version
    end
    return installs
end

# Modified main menu function
function main_menu()
    while true
        try
            render_menu_header()
            println(color("\nSelect a simulation mode:", :blue))
            render_menu_options()

            print("\n" * color("→ ", :green))
            choice = lowercase(strip(readline()))

            if choice in ["1", "2", "3"]
                run_simulation(parse(Int, choice))
                print("\nPress Enter to continue...")
                readline()
            elseif choice == "p"
                check_and_install_packages()
            elseif choice == "h"
                show_help()
            elseif choice == "q"
                println(color("\nThank you for using the Particle Motion Simulator!", :cyan))
                break
            else
                println(color("\n⚠ Invalid choice. Please try again.", :red))
                sleep(1)
            end
        catch e
            println(color("\n⚠ Error: $(sprint(showerror, e))", :red))
            sleep(2)
        end
    end
end

# Start the application
function start_simulator()
    try
        main_menu()
    catch e
        println(color("\n⚠ Fatal Error: $(sprint(showerror, e))", :red))
        println(color("The application will now exit.", :red))
        return
    end
end

# Run the simulator
start_simulator()
