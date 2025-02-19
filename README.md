# Particle Motion Simulation in Electromagnetic Fields

[![DOI](https://zenodo.org/badge/841295299.svg)](https://doi.org/10.5281/zenodo.14451939)

This repository provides a modern and professional framework for simulating particle motion in electromagnetic fields. It leverages **Julia** for high-performance numerical simulations and **Python** for advanced visualization and data analysis. The project offers a streamlined workflow, from running simulations via a user-friendly terminal interface to analyzing results with detailed plots and visualizations.

## Table of Contents

1. [Key Features](#key-features)
2. [Installation](#installation)
    1. [Prerequisites](#prerequisites)
    2. [Setup](#setup)
3. [Usage](#usage)
    1. [Running Simulations](#running-simulations)
    2. [Visualization and Post-Processing](#visualization-and-post-processing)
4. [Configuration](#configuration)
5. [Physics and Equations of Motion](#physics-and-equations-of-motion)
6. [Appendix](#appendix)
7. [Contributors](#contributors)

## Key Features

- **High-Performance Simulations**: Utilizes Julia for efficient numerical computations.
- **Interactive Visualization**: Python-based tools for generating 2D and 3D plots, energy and momentum conservation analysis, and peak detection.
- **User-Friendly Interface**: A menu-driven terminal interface for running simulations and configuring parameters.
- **Comprehensive Simulation Modes**: Supports 1D, 2D, and 3D simulations in cylindrical coordinates.
- **Adiabatic Invariants Analysis**: Tools for detecting extrema and analyzing invariants in particle motion.
- **Customizable Initial Conditions**: Easily modify initial parameters without changing source code.

## Installation

### Prerequisites

Ensure the following software is installed on your system:

- **Julia**: [Download Julia](https://julialang.org/downloads/)
- **Python**: [Download Python](https://www.python.org/downloads/)
- **Pip**: [Install Pip](https://pip.pypa.io/en/stable/installation/)

### Setup

#### Method 1: Automated Setup

Use the provided Julia and Python scripts for automated package management by selecting the **Package Management**:

```bash
julia Start.jl
```

and when you run the `visualization.py` it automatically check and install all requirements:

```bash
python3 visualization.py
```

#### Method 2: Manual Setup

Alternatively, install dependencies manually:

1. **Julia Packages**:

    ```bash
    julia install_packages.jl
    ```

2. **Python Packages**:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Usage

### Running Simulations

1. **Launch the Simulator**:

    ```bash
    julia Start.jl
    ```

2. **Select Simulation Mode**:
    - [1] 1D Simulation
    - [2] 2D Simulation
    - [3] 3D Simulation

3. **Modify Initial Conditions**:
    Update the `initial_conditions.txt` file to set parameters such as:

    ```plaintext
    theta_0 = π/60
    alpha_0 = π/2
    beta_0 = 0
    phi_0 = 0
    epsilon = 0.002
    eps_phi = 0.0
    kappa = 0.01
    delta_star = 0.01
    time_end = 300.0
    ```

4. **Run Simulations**:
    Monitor progress and receive completion feedback in the terminal.

**Screen Recording**:

[![Video Thumbnail](https://via.placeholder.com/640x360.png?text=Click+to+Watch+Video)](https://github.com/user-attachments/assets/b58eab73-5906-48b0-a356-a0eec0d89a57)

### Visualization and Post-Processing

After running simulations, use the `visualization.py` script for all visualization tasks. The available options include:

- **Unified Terminal Interface**: Run 1D, 2D, and 3D simulations directly from a Julia-based menu-driven UI.
- **Interactive Parameter Configuration**: Modify initial conditions and simulation settings without changing source code.
- **Visualization Tools**: Python scripts for generating 2D and 3D plots, analyzing energy and momentum conservation, and detecting peaks in particle trajectories.
- **Comprehensive Simulation Modes**:
  - **1D Simulation**: Particle motion along a single axis.
  - **2D Simulation**: Particle motion in a plane with radial and vertical dynamics.
  - **3D Simulation**: Full spatial dynamics in cylindrical coordinates.
- **Peak Detection**: Identify extrema and analyze adiabatic invariants in particle motion.
- **Energy and Momentum Conservation**: Visualize energy and momentum trends over time.

Example usage:

```bash
python3 visualization.py
```

Follow the interactive prompts to select the desired analysis or plotting mode.

## Configuration

### Initial Conditions

Set initial conditions in the `initial_conditions.txt` file. Key parameters include:

- **theta_0**: Initial trajectory angle in radians.
- **alpha_0**: Initial pitch angle in radians.
- **beta_0**: Initial azimuthal angle in radians.
- **phi_0**: Initial azimuthal position in radians.
- **epsilon**: Dimensionless parameter representing the initial velocity magnitude.
- **eps_phi**: Electric field perturbation parameter.
- **kappa**: Magnetic field gradient strength.
- **delta_star**: Drift correction factor.
- **time_end**: Total simulation duration.

## Physics and Equations of Motion

### 2D Simulation

The 2D simulation models particle motion in **cylindrical coordinates** (`ρ`, `z`, `φ`). The key equations solved include:

- **Equation of Motion for `ρ` and `z`**:

```math
\frac{d^2 \rho}{dt^2}, \frac{d^2 z}{dt^2}
```

These describe the particle's motion in radial and vertical directions under the influence of electromagnetic fields.

### 3D Simulation

The 3D simulation solves for **radial distance** (`ρ`), **vertical position** (`z`), and **azimuthal angle** (`φ`). The key equations of motion involve:

```math
\frac{d{\widetilde{R}}^2}{{d\tau}^2}=\frac{ \widetilde{R}.\widetilde{z}}{{(\widetilde{R}^2+\widetilde{z}^2)}^{3/2}} \frac{d\phi}{d\tau}+ \widetilde{R} ({\frac{d\phi}{d\tau}})^2-\epsilon_\phi\frac{\partial \widetilde{\Phi}}{\partial \widetilde{R}}
```

```math
\frac{d{\widetilde{\phi}}^2}{{d\tau}^2}=\space(\frac{1}{{(\widetilde{R}^2+\widetilde{z}^2)}^{3/2}} (\widetilde{R}\frac{d\widetilde{z}}{d\tau}-\widetilde{z}\frac{d\widetilde{R}}{d\tau})- 2\frac{d\widetilde{R}}{d\tau}\frac{d\phi}{d\tau})/\widetilde{R}
```

```math
\frac{d{\widetilde{z}}^2}{{d\tau}^2}=- \frac{\widetilde{R}^2}{{(\widetilde{R}^2+\widetilde{z}^2)}^{3/2}} \frac{d\phi}{d\tau}-\epsilon_\phi\frac{\partial \widetilde{\Phi}}{\partial \widetilde{z}}
```

This simulates the full trajectory in a 3D cylindrical coordinate system under magnetic and electric fields. The guiding center approximation can also be employed for specific scenarios.

- **Electric Potential Function**:

   ```math
    \Phi = \kappa  \Phi _0 \left(1-\delta_*^2 \left(1-\frac{z^2}{R^2+z^2}\right)\right) \ln \left(\frac{1}{R^2+z^2}\right)+\frac{1}{2} \phi _0 \left(1-\frac{z^2}{R^2+z^2}\right)
  ```

- **Expanded Derivatives**:

   Radial derivative:

    ```math
  \frac{\partial \Phi}{\partial R}=\frac{R \Phi _0 \left(-2 \kappa  R^2+2 \delta_*^2 \kappa  \left(R^2-z^2 \log
   \left(\frac{1}{R^2+z^2}\right)\right)+(1-2 \kappa )
   z^2\right)}{\left(R^2+z^2\right)^2}
    ```

   Vertical derivative:

    ```math
  \frac{\partial \Phi}{\partial z}=-\frac{z \Phi _0 \left((2 \kappa +1) R^2-2 \delta_*^2 \kappa  R^2 \left(\log
   \left(\frac{1}{R^2+z^2}\right)+1\right)+2 \kappa 
   z^2\right)}{\left(R^2+z^2\right)^2}
    ```

These equations describe the magnetic flux function and its derivatives in the radial and vertical directions, which are crucial for understanding the magnetic field structure in the simulation.

## Appendix

For additional information, refer to the documentation and examples provided in the repository.

## Contributors

- Shahab Bahreini Jangjoo
