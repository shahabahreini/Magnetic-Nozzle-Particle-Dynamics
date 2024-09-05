# adiabatic_invariant_julia 🧲

This project offers a numerical simulation of charged particle dynamics in EM (Electromagnetic) fields. It strives to
model the magnetic nozzle structure and helps develop a stable, low-error method to preserve total energy and
investigate the adiabatic invariant. This is part of an M.Sc. Thesis at the University of Saskatchewan - Particle in EM
Fields, in the Department of Computer Science.

## Installation Requirements 🛠️

Follow these instructions to get your system up and running:

### 1. Install Julia 🚀

Visit the [official website](https://julialang.org/downloads/) and follow the provided instructions to install Julia.

### 2. Install Python and Pip 🐍

- **Python:** Use the [Python official instructions](https://www.python.org/downloads/) to install Python on your OS.
- **Pip:** To install Pip (needed for package installation), follow
  the [instructions here](https://pip.pypa.io/en/stable/installation/).

### 3. Install Julia and Python Packages 📦

In the root folder, run the following commands:

```bash
julia install_packages.jl
python3 pip install -r requirements.txt
```

## Running Simulation 🚀

### Updating Initial Conditions

Edit "**initial_conditions.txt**" to update the simulation's initial conditions.

### Updating Simulation Equations

Modify the functions "**CylindricalProblem!**" in "**Simulation2D.jl**" or "**Simulation3D.jl**" to change the dynamics
of the simulations.

### Numerical Simulation 🧮

Run the Julia core using one of the following methods:

a) **Single-Core Mode:** Use this command for single-core processing, then follow the on-screen instructions:

```
julia Start.jl
```

b) **Multi-Core Mode:** For multi-core processing, add '**-t [number of cores]**' before calling the Julia script.
Here's an example using 4 cores:

```
julia -t 4 Start.jl
```

### Plotting The Results

For 2D simulations please use:

```
python3 plotter_2D.py
```

And for 3D simulations please use:

```
python3 plotter_3D.py
```

---

Happy Simulating! Feel free to contribute, report issues, or ask questions.
