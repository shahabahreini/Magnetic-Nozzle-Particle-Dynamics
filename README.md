
# Particle Dynamics in Electromagnetic Fields Simulation

This project involves the numerical simulation of particle dynamics and properties in electromagnetic fields, using Julia for the core numerical computations and Python for visualization. The simulation tracks particle motion under specific initial conditions and renders both 2D and 3D plots to visualize the system's behavior over time.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Initial Conditions](#initial-conditions)
  - [Running the Simulations](#running-the-simulations)
  - [Visualization](#visualization)
- [Plotting Options](#plotting-options)
  - [2D Plotting Options](#2d-plotting-options)
  - [3D Plotting Options](#3d-plotting-options)
- [Configuration](#configuration)
- [Contributors](#contributors)

## Project Overview

This simulation project models the behavior of particles in electromagnetic fields. The simulation integrates the particle's equations of motion over time and visualizes the trajectory in both 2D and 3D spaces. It supports:

- **Particle trajectory computation** in an electromagnetic field.
- **Energy and momentum conservation** visualizations.
- **2D and 3D plot generation** using Python libraries like Matplotlib and Plotly.
- **Analysis of peaks** in particle motion data.


## Installation

### 1. Install Python, Julia and Pip

- **Python:** Use the [Python official instructions](https://www.python.org/downloads/) to install Python on your OS.
- **Pip:** To install Pip (needed for package installation), follow
  the [instructions here](https://pip.pypa.io/en/stable/installation/).
- **Julia** Please follow the instruction in [Julia Official Download Page](https://julialang.org/downloads/).

### 2. Install Julia and Python Packages 📦

Ensure Julia is installed as it is used for the core numerical simulations, while Python is used for post-processing and visualization.
In the root folder, run the following commands:

```bash
julia install_packages.jl
python3 pip install -r requirements.txt
```

## Usage

### Initial Conditions

Modify the system's initial conditions in the `initial_conditions.txt` file. The default conditions are:

```
theta_0 = π/10
alpha_0 = π/2 
beta_0 = 0
phi_0 = π/2
epsilon = 0.0005
eps_phi = 0.0
kappa = 0.0
delta_star = 1.0
number_of_trajectories = 10000
time_end = 50000.0
```

These parameters control the simulation's behavior and can be adjusted as needed.

### Running the Simulations

1. Ensure the Julia simulation script is configured correctly and generates the trajectory data.
2. Use Python scripts to process and visualize the data:

- **2D Plotting**:
    ```bash
    python plotter_2D.py
    ```

- **3D Plotting**:
    ```bash
    python plotter_3D.py
    ```

- **Energy and Momentum Visualization**:
    ```bash
    python plotter_energy_momentum_conservasion.py
    ```

- **Peak Finder**:
    ```bash
    python peakfinder.py
    ```

### Visualization

The Python scripts use Matplotlib, Plotly, and Seaborn for plotting. The following visualizations are supported:

- **2D Particle Trajectories**: Generated with `plotter_2D.py`.
- **3D Particle Trajectories**: Created with `plotter_3D.py`.
- **Energy and Momentum Conservation**: Tracked with `plotter_energy_momentum_conservasion.py`.

## Plotting Options

### 2D Plotting Options (`plotter_2D.py`)

- **Trajectory Plot**: The script generates 2D projections of particle trajectories over time. You can customize the time range, axis labels, and line styles.
- **Phase Space Plot**: Visualizes particle position vs. velocity for phase-space analysis.
- **Custom Axes**: Allows users to set axis ranges, labels, and gridlines.
- **Extremum Detection**: Use the `show_extremums_peaks` option to mark peaks in specified variables (e.g., density).

### 3D Plotting Options (`plotter_3D.py`)

- **3D Trajectory Plot**: Plots the particle's trajectory in 3D space. Rotate, pan, and zoom in on sections for detailed analysis.
- **Interactive Plotting**: Built using Plotly, allowing users to interact with the plot (zoom, pan, rotate).
- **Custom Colors and Themes**: Customize the colors and themes to highlight specific features.
- **Subsampling**: For large datasets, subsampling reduces visual clutter while retaining the overall trajectory shape.
- **Guiding Center Approximation**: If `based_on_guiding_center` is set to `true`, the guiding center approximation is used.

## Configuration

The `config.yaml` file contains settings for file paths, output formats, and simulation parameters. Key options:

- **save_file_name**: Base name for the output file.
- **save_file_extension**: File format for saved plots (e.g., `.svg` or `.png`).
- **is_multi_files**: Set to `true` to process multiple CSV files.
- **based_on_guiding_center**: If `true`, uses the guiding center approximation.
- **simulation_parameters**: Control physical properties like `eps`, `epsphi`, `kappa`, `beta`, `alpha`, `theta`, and simulation time.
- **method**: Numerical method for the simulation (e.g., `"Feagin14 Method"`).