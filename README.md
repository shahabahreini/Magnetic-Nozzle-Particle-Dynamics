
# Particle Dynamics in Electromagnetic Fields Simulation

This project involves the numerical simulation of particle dynamics and properties in electromagnetic fields, using Julia for the core numerical computations and Python for visualization. The simulation tracks particle motion under specific initial conditions and renders both 2D and 3D plots to visualize the system's behavior over time.

## Table of Contents

<!-- TOC -->

- [1. Project Overview](#1-project-overview)
- [2. Installation](#2-installation)
  - [2.1. Prerequisites](#21-prerequisites)
  - [2.2. Package Installation](#22-package-installation)
- [3. Usage](#3-usage)
  - [3.1. Initial Conditions](#31-initial-conditions)
  - [3.2. Running the Simulations](#32-running-the-simulations)
    - [3.2.1. Julia Simulation](#321-julia-simulation)
    - [3.2.2. Python Data Processing and Visualization](#322-python-data-processing-and-visualization)
  - [3.3. Visualization](#33-visualization)
- [4. Plotting Options](#4-plotting-options)
  - [4.1. 2D Plotting Options](#41-2d-plotting-options)
  - [4.2. 3D Plotting Options](#42-3d-plotting-options)
- [5. Configuration](#5-configuration)
- [6. Additional Features](#6-additional-features)
  - [6.1. Peak Finder Script](#61-peak-finder-script)
    - [6.1.1. Features](#611-features)
    - [6.1.2. Configuration](#612-configuration)
    - [6.1.3. Usage](#613-usage)
    - [6.1.4. Example Workflow](#614-example-workflow)
    - [6.1.5. Output](#615-output)
- [7. Contributors](#7-contributors)

<!-- /TOC -->

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

#### Step 1: Running the Julia Simulation

To start the simulation, ensure that the necessary Julia scripts (`Simulation2D.jl` and `Simulation3D.jl`) are available and configured. You can choose between a 2D or 3D simulation:

1. Open a terminal or command prompt.
2. Navigate to the directory where `Start.jl` and the associated simulation files are located.
3. Run the simulation using the following command:

    ```bash
    julia Start.jl
    ```

4. Choose the type of simulation when prompted:
   - **1** for the 3D simulation
   - **2** for the 2D simulation
   - **n** to exit the program

5. After the simulation completes, you can modify the initial conditions in the respective simulation files if needed and run the simulation again.

#### Step 2: Processing and Visualizing the Data with Python

Once the simulation is complete, use Python scripts to process the generated trajectory data and visualize the results:

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

The Python scripts utilize Matplotlib, Plotly, and Seaborn for generating visualizations. The following types of plots are supported:

- **2D Particle Trajectories**: Generated using the `plotter_2D.py` script.
- **3D Particle Trajectories**: Created using the `plotter_3D.py` script.
- **Energy and Momentum Conservation**: Visualized using the `plotter_energy_momentum_conservasion.py` script.
- **Peak Detection**: Analyzed and visualized with the `peakfinder.py` script, showing critical points in the particle motion data.


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

## Additional info
### Peak Finder Script (`peakfinder.py`)

The `peakfinder.py` script is designed to detect peaks and analyze specific variables from simulation datasets. It allows users to visualize critical points in the data and calculate additional physical quantities like adiabatic invariants and magnetic moments. The script processes data stored in CSV files and generates plots based on the detected peaks.

#### Features:
- **Peak Detection**: Automatically identifies local maxima (peaks) in a chosen variable using the `findpeaks` library and SciPy.
- **Multi-file Support**: The script can process multiple CSV files at once if enabled.
- **Customizable Plotting**: Allows customization of plot titles, axis labels, and peak markers.
- **Adiabatic Calculations**: Calculates and visualizes adiabatic invariants if configured.
- **Magnetic Moment**: Computes and plots the traditional magnetic moment.

#### Configuration:
The script reads parameters from the `config.yaml` file to control its behavior. Key configuration options include:

- **save_file_name**: The base name for output plot files.
- **save_file_extension**: The file extension for output plots (e.g., `.svg`, `.png`).
- **is_multi_files**: Set this to `true` to process multiple CSV files, or `false` for a single file.
- **target_folder_multi_files**: Directory where multiple CSV files are stored for processing.
- **plots_folder**: Directory where generated plots will be saved.
- **extremum_of**: The variable to analyze for peak detection (e.g., `"drho"` for density changes).
- **based_on_guiding_center**: If set to `true`, the guiding center approximation is used for calculations.
- **calculate_integral**: Set to `true` to calculate and plot adiabatic invariants.
- **calculate_traditional_magneticMoment**: Set to `true` to compute and plot the magnetic moment.

#### Usage:
To run the script, ensure the necessary Python libraries are installed and configured according to `requirements.txt`. The script processes data and generates plots as follows:

1. **Single File Processing**:
   If `is_multi_files` is set to `false`, the script will process a single CSV file. The file path can be selected interactively or hardcoded.

    ```bash
    python peakfinder.py
    ```

2. **Multiple File Processing**:
   If `is_multi_files` is set to `true`, the script will process all CSV files in the directory specified by `target_folder_multi_files`.

    ```bash
    python peakfinder.py
    ```

#### Example Workflow:
1. **Peak Detection**: The script detects peaks in the variable specified by `extremum_of` (e.g., `"drho"`) and visualizes the data with peak markers.
2. **Adiabatic Invariant Calculation**: If enabled, the script computes the adiabatic invariant \( J = \oint v_x \, dx \) and plots it as a function of time or other parameters.
3. **Magnetic Moment Calculation**: If `calculate_traditional_magneticMoment` is enabled, the traditional magnetic moment is computed and visualized.

#### Output:
- The results are saved as plot files in the format specified by `save_file_extension`.
- The plots include annotations for detected peaks, and additional calculations (e.g., adiabatic invariant or magnetic moment) are also visualized.
