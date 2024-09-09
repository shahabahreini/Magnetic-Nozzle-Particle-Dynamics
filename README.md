# Particle Dynamics in Electromagnetic Fields Simulation

This project involves the numerical simulation of particle dynamics and properties in electromagnetic fields, using Julia for the core numerical computations and Python for visualization. The simulation tracks particle motion under specific initial conditions and renders both 2D and 3D plots to visualize the system's behavior over time.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Installation](#2-installation)
   1. [Prerequisites](#21-prerequisites)
   2. [Package Installation](#22-package-installation)
3. [Usage](#3-usage)
   1. [Initial Conditions](#31-initial-conditions)
   2. [Running the Simulations](#32-running-the-simulations)
      1. [Julia Simulation](#321-julia-simulation)
      2. [Python Data Processing and Visualization](#322-python-data-processing-and-visualization)
   3. [Visualization](#33-visualization)
4. [Plotting Options](#4-plotting-options)
   1. [2D Plotting Options](#41-2d-plotting-options-plotter_2dpy)
   2. [3D Plotting Options](#42-3d-plotting-options-plotter_3dpy)
5. [Configuration](#5-configuration)
6. [Additional Features](#6-additional-features)
   1. [Peak Finder Script](#61-peak-finder-script-peakfinderpy)
      1. [Features](#611-features)
      2. [Configuration](#612-configuration)
      3. [Usage](#613-usage)
      4. [Example Workflow](#614-example-workflow)
      5. [Output](#615-output)
7. [Contributors](#7-contributors)

## 1. Project Overview

This simulation project models the behavior of particles in electromagnetic fields. The simulation integrates the particle's equations of motion over time and visualizes the trajectory in both 2D and 3D spaces. It supports:

- **Particle trajectory computation** in an electromagnetic field.
- **Energy and momentum conservation** visualizations.
- **2D and 3D plot generation** using Python libraries like Matplotlib and Plotly.
- **Analysis of peaks** in particle motion data.

## 2. Installation

### 2.1. Prerequisites

- **Python:** Use the [Python official instructions](https://www.python.org/downloads/) to install Python on your OS.
- **Pip:** To install Pip (needed for package installation), follow the [instructions here](https://pip.pypa.io/en/stable/installation/).
- **Julia:** Please follow the instruction in [Julia Official Download Page](https://julialang.org/downloads/).

### 2.2. Package Installation

Ensure Julia is installed as it is used for the core numerical simulations, while Python is used for post-processing and visualization.
In the root folder, run the following commands:

```bash
julia install_packages.jl
python3 -m pip install -r requirements.txt
```

## 3. Usage

### 3.1. Initial Conditions

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

### 3.2. Running the Simulations

#### 3.2.1. Julia Simulation

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

#### 3.2.2. Python Data Processing and Visualization

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

### 3.3. Visualization

The Python scripts utilize Matplotlib, Plotly, and Seaborn for generating visualizations. The following types of plots are supported:

- **2D Particle Trajectories**: Generated using the `plotter_2D.py` script.
- **3D Particle Trajectories**: Created using the `plotter_3D.py` script.
- **Energy and Momentum Conservation**: Visualized using the `plotter_energy_momentum_conservasion.py` script.
- **Peak Detection**: Analyzed and visualized with the `peakfinder.py` script, showing critical points in the particle motion data.

## 4. Plotting Options

### 4.1. 2D Plotting Options (`plotter_2D.py`)

- **Trajectory Plot**: The script generates 2D projections of particle trajectories over time. You can customize the time range, axis labels, and line styles.
- **Phase Space Plot**: Visualizes particle position vs. velocity for phase-space analysis.
- **Custom Axes**: Allows users to set axis ranges, labels, and gridlines.
- **Extremum Detection**: Use the `show_extremums_peaks` option to mark peaks in specified variables (e.g., density).

### 4.2. 3D Plotting Options (`plotter_3D.py`)

- **3D Trajectory Plot**: Plots the particle's trajectory in 3D space. Rotate, pan, and zoom in on sections for detailed analysis.
- **Interactive Plotting**: Built using Plotly, allowing users to interact with the plot (zoom, pan, rotate).
- **Custom Colors and Themes**: Customize the colors and themes to highlight specific features.
- **Subsampling**: For large datasets, subsampling reduces visual clutter while retaining the overall trajectory shape.
- **Guiding Center Approximation**: If `based_on_guiding_center` is set to `true`, the guiding center approximation is used.

## 5. Configuration

The `config.yaml` file contains settings for file paths, output formats, and simulation parameters. Key options:

- **save_file_name**: Base name for the output file.
- **save_file_extension**: File format for saved plots (e.g., `.svg` or `.png`).
- **is_multi_files**: Set to `true` to process multiple CSV files.
- **based_on_guiding_center**: If `true`, uses the guiding center approximation.
- **simulation_parameters**: Control physical properties like `eps`, `epsphi`, `kappa`, `beta`, `alpha`, `theta`, and simulation time.
- **method**: Numerical method for the simulation (e.g., `"Feagin14 Method"`).

## 6. Additional Features

### 6.1. Peak Finder Script (`peakfinder.py`)

The `peakfinder.py` script is designed to detect peaks and analyze specific variables from simulation datasets. It allows users to visualize critical points in the data and calculate additional physical quantities like adiabatic invariants and magnetic moments. The script processes data stored in CSV files and generates plots based on the detected peaks.

#### 6.1.1. Features:
- **Peak Detection**: Automatically identifies local maxima (peaks) in a chosen variable using the `findpeaks` library and SciPy.
- **Multi-file Support**: The script can process multiple CSV files at once if enabled.
- **Customizable Plotting**: Allows customization of plot titles, axis labels, and peak markers.
- **Adiabatic Calculations**: Calculates and visualizes adiabatic invariants if configured.
- **Magnetic Moment**: Computes and plots the traditional magnetic moment.

#### 6.1.2. Configuration:
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

#### 6.1.3. Usage:
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

#### 6.1.4. Example Workflow:
1. **Peak Detection**: The script detects peaks in the variable specified by `extremum_of` (e.g., `"drho"`) and visualizes the data with peak markers.
2. **Adiabatic Invariant Calculation**: If enabled, the script computes the adiabatic invariant \( J = \oint v_x \, dx \) and plots it as a function of time or other parameters.
3. **Magnetic Moment Calculation**: If `calculate_traditional_magneticMoment` is enabled, the traditional magnetic moment is computed and visualized.

#### 6.1.5. Output:
- The results are saved as plot files in the format specified by `save_file_extension` in the `plots_folder` directory.
- Each plot is labeled with relevant information such as the variable analyzed, peak locations, and any calculated quantities.

## 7. Contributors

[List of contributors or link to contributors file]



### Updated 2D Plotting Options (`plotter_2D.py`)

The `plotter_2D.py` script generates 2D plots from the output CSV files produced by the simulation. It supports both single-file and multi-file mode with detailed control over axis labels, titles, legends, and more.

#### Features:
- **Single File Mode:** Plot data from a single CSV file.
- **Multi-file Mode:** Plot data from multiple CSV files simultaneously with options for grouping and sorting by specific parameters (e.g., `eps`).
- **Dynamic Titles and Legends:** Titles, axis labels, and legends are automatically generated using LaTeX symbols for the parameters, ensuring readability and consistency across different plots.
- **Sorted Legends:** In multi-file mode, the legend is automatically sorted based on a specific parameter (e.g., `eps`).
- **Interactive File Selection:** Users can select files from the current directory or a specific folder to plot.
- **Custom Axis Labels:** The x and y axes are labeled using parameters such as `rho`, `z`, `drho`, `dz`, `omega_rho`, and `omega_z`, all represented in proper scientific notation.

#### Usage:
To use the updated `plotter_2D.py`, follow these steps:

1. **Single File Mode**:
   Run the script and select a single CSV file from the current directory:
   ```bash
   python plotter_2D.py
   ```
   You will be prompted to select the x-axis and y-axis parameters from the available options.

2. **Multi-file Mode**:
   To plot multiple CSV files, you can either:
   - Select files from a specific folder.
   - Select all files in the current directory for comparison.
   
   When using multi-file mode, the legend will be grouped and sorted by the most varying parameter (e.g., `eps`), and only varying parameters will be shown in the legend. The common parameters will be included in the plot title.

3. **Progress Bar**:
   A progress bar is displayed while the script reads data, generates the plot, saves it to the `plots` folder, and finally shows the plot.

#### Plot Customization:
- **Title and Legend**: The script automatically generates titles and legends using LaTeX symbols for the parameters. The common parameters are shown in the plot title, while varying parameters are displayed in the legend.
- **Sorting Legends**: In multi-file mode, the legend is sorted by the parameter `eps` (or another selected parameter), ensuring consistency in the order of the plots.
- **Saving the Plot**: The plot is saved in the `plots` folder with a filename format `[singlemode/multimode]-YYYY-MM-DD-HH-MM.png`.

#### Example:
1. **Single File Plot**:
   ```bash
   python plotter_2D.py
   ```
   Select file `3D_export-eps0.001-epsphi0.0-kappa0.0.csv`, x-axis as `tau` (time), and y-axis as `rho` (position).

   - **Flow**: 
     1. The user runs the command and selects the single-file option.
     2. The user picks a CSV file from the list.
     3. The script asks for x and y-axis selections (e.g., `tau` and `rho`).
     4. A progress bar appears as the plot is generated and saved.
     5. The final plot is shown, and the plot is saved in the `plots` folder.

2. **Multi-file Plot**:
   ```bash
   python plotter_2D.py
   ```
   Select multiple files from the `current folder` option. The script will group files by shared parameters (e.g., `kappa`, `beta`, etc.) and sort the plots by `eps`.

   - **Flow**:
     1. The user selects the multi-file option.
     2. Files are selected from the specified folder or current directory.
     3. The user picks x and y-axis parameters.
     4. The progress bar tracks the plotting process.
     5. The plot is generated, with the legend sorted by `eps` or another varying parameter, and saved in the `plots` folder.

#### Available Parameters:
- **x-axis and y-axis options**:
  - `rho`: Radial distance (`$	ilde{R}$`)
  - `z`: Vertical distance (`$	ilde{Z}$`)
  - `drho`: Radial velocity (`$d	ilde{R}/d	au$`)
  - `dz`: Vertical velocity (`$d	ilde{Z}/d	au$`)
  - `omega_rho`: Radial frequency (`$\omega_{	ilde{R}}$`)
  - `omega_z`: Vertical frequency (`$\omega_{	ilde{Z}}$`)
  - `timestamp`: Time (`$	au$`)

- **Common Parameters Included in Titles**:
  - `eps`: Perturbation strength (`$\epsilon$`)
  - `epsphi`: Perturbation phase (`$\epsilon_{\phi}$`)
  - `kappa`: Curvature (`$\kappa$`)
  - `deltas`: Delta shift (`$\delta_s$`)
  - `beta`: Magnetic beta (`$eta$`)
  - `alpha`: Magnetic inclination (`$lpha$`)
  - `theta`: Magnetic azimuthal angle (`$	heta$`)
  - `time`: Total simulation time (`$	au$`)

#### Notes:
- Ensure that the CSV files generated by the simulation contain the necessary columns for the parameters you wish to plot.
- For large datasets, the script subsamples the data points to maintain plot clarity.

#### Example Output:
Below is an example of a generated plot in multi-file mode where multiple CSV files were plotted and sorted by `eps`. The common parameters were placed in the plot title, and the varying `eps` values were shown in the legend.

![Example Plot](path/to/plot_image.png)

