import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal output
init()

# Configuration Section
# =====================

# Plot parameters
PLOT_CONFIG = {
    'dpi': 300,                      # Resolution of the saved plots (dots per inch)
    'font_size': 12,                 # Base font size for plot text
    'legend_font_size': 10,          # Font size for legend text
    'tick_label_size': 10,           # Font size for axis tick labels
    'figure_size': (12, 10),         # Size of the figure in inches (width, height)
    'colorbar_pad': 0.1,             # Padding between plots and colorbars
    'output_folder': os.path.join('Plots', 'EField_Plots'),  # Folder to save plots
    'show_plots': False,             # Set to True to display plots on screen after saving
}

# Physical constants
PHYSICS_PARAMS = {
    'B0': 1.0,      # Magnetic field strength at the magnetic axis
    'r0': 1.0,      # Reference radius
    'Phi0': 1.0,    # Reference electrostatic potential
    'kappa': 1.0,   # Temperature gradient parameter
    'r_star': 2.0,  # Characteristic radius for temperature profile
    'n0': 1.0,      # Reference density
}

# Plot-specific parameters
PLOT_PARAMS = {
    '3d_quiver': {
        'grid_size': 15,         # Number of grid points in each dimension
        'R_range': (0.1, 5),     # Range of R values (min, max)
        'Z_range': (0.1, 5),     # Range of Z values (min, max)
        'quiver_length': 0.5,    # Length of quiver arrows
    },
    '3d_contour': {
        'grid_size': 100,        # Number of grid points in each dimension
        'R_range': (0.1, 5),     # Range of R values (min, max)
        'Z_range': (0.1, 5),     # Range of Z values (min, max)
        'contour_levels': 20,    # Number of contour levels
        'view_angle': (30, 45),  # View angle for 3D plot (elevation, azimuth)
    },
    '2d_streamplot': {
        'grid_size': 100,        # Number of grid points in each dimension
        'R_range': (0.1, 5),     # Range of R values (min, max)
        'Z_range': (0.1, 5),     # Range of Z values (min, max)
        'density': 1.5,          # Density of streamlines
        'linewidth': 1.5,        # Width of streamlines
        'arrowsize': 1.2,        # Size of arrows on streamlines
    },
    '2d_contour': {
        'grid_size': 100,        # Number of grid points in each dimension
        'R_range': (0.1, 5),     # Range of R values (min, max)
        'Z_range': (0.1, 5),     # Range of Z values (min, max)
        'contour_levels': 50,    # Number of contour levels
    },
}

# Helper functions
def create_directory(path):
    """Create a directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"{Fore.GREEN}✔ Created directory: {path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}✘ Error creating directory {path}: {e}{Style.RESET_ALL}")
        sys.exit(1)

def print_section(title):
    """Print a formatted section title."""
    print(f"\n{Fore.CYAN}{'=' * 40}")
    print(f"{title:^40}")
    print(f"{'=' * 40}{Style.RESET_ALL}")

# Create output folders
create_directory(PLOT_CONFIG['output_folder'])

# Set up high-quality plot parameters
plt.rcParams['figure.dpi'] = PLOT_CONFIG['dpi']
plt.rcParams['savefig.dpi'] = PLOT_CONFIG['dpi']
plt.rcParams['font.size'] = PLOT_CONFIG['font_size']
plt.rcParams['legend.fontsize'] = PLOT_CONFIG['legend_font_size']
plt.rcParams['xtick.labelsize'] = PLOT_CONFIG['tick_label_size']
plt.rcParams['ytick.labelsize'] = PLOT_CONFIG['tick_label_size']

# Functions
def psi(R, z):
    return PHYSICS_PARAMS['B0'] * PHYSICS_PARAMS['r0']**2 * (1 - z / np.sqrt(R**2 + z**2))

def varsigma(R, z):
    return 1 - (1 - psi(R, z) / (PHYSICS_PARAMS['B0'] * PHYSICS_PARAMS['r0']**2))**2

def Phi_star(R, z):
    return 0.5 * PHYSICS_PARAMS['Phi0'] * varsigma(R, z)

def T_psi(R, z):
    return PHYSICS_PARAMS['kappa'] * PHYSICS_PARAMS['Phi0'] * (1 - (PHYSICS_PARAMS['r0']**2 / PHYSICS_PARAMS['r_star']**2) * varsigma(R, z))

def n(R, z):
    return PHYSICS_PARAMS['n0'] * np.exp((Phi_star(R, z) - PHYSICS_PARAMS['Phi0']) / T_psi(R, z))

def E_R(R, z):
    dR = 1e-6
    return -(Phi_star(R + dR, z) - Phi_star(R, z)) / dR - (T_psi(R, z) / n(R, z)) * (n(R + dR, z) - n(R, z)) / dR

def E_z(R, z):
    dz = 1e-6
    return -(Phi_star(R, z + dz) - Phi_star(R, z)) / dz - (T_psi(R, z) / n(R, z)) * (n(R, z + dz) - n(R, z)) / dz

# Plotting functions
def create_plot(plot_type, plot_func):
    """Create and save a plot with error handling."""
    print_section(f"Creating {plot_type} plot")
    try:
        plot_func()
        print(f"{Fore.GREEN}✔ {plot_type} plot created successfully{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}✘ Error creating {plot_type} plot: {e}{Style.RESET_ALL}")

def save_plot(fig, plot_type):
    """Save the plot with a timestamp in the filename and optionally display it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{plot_type}_{timestamp}.png"
    filepath = os.path.join(PLOT_CONFIG['output_folder'], filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"{Fore.GREEN}✔ Plot saved: {filepath}{Style.RESET_ALL}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close(fig)

def create_3d_quiver_plot():
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size'])
    ax = fig.add_subplot(111, projection='3d')

    R = np.linspace(*PLOT_PARAMS['3d_quiver']['R_range'], PLOT_PARAMS['3d_quiver']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['3d_quiver']['Z_range'], PLOT_PARAMS['3d_quiver']['grid_size'])
    R, Z = np.meshgrid(R, Z)

    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    E_magnitude = np.sqrt(E_R_values**2 + E_z_values**2)

    quiver = ax.quiver(R, Z, np.zeros_like(R), E_R_values, E_z_values, np.zeros_like(R), 
                       length=PLOT_PARAMS['3d_quiver']['quiver_length'], normalize=True, cmap='viridis')

    cbar = plt.colorbar(quiver, ax=ax, label='Electric field magnitude', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title('Electric Field Distribution (3D Quiver)')

    plt.tight_layout()
    save_plot(fig, '3d_quiver')

def create_3d_contour_plot():
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size'])
    ax = fig.add_subplot(111, projection='3d')

    R = np.linspace(*PLOT_PARAMS['3d_contour']['R_range'], PLOT_PARAMS['3d_contour']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['3d_contour']['Z_range'], PLOT_PARAMS['3d_contour']['grid_size'])
    R, Z = np.meshgrid(R, Z)

    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    E_magnitude = np.sqrt(E_R_values**2 + E_z_values**2)

    surf = ax.plot_surface(R, Z, E_magnitude, cmap='viridis', edgecolor='none', alpha=0.8)

    contour_levels = np.linspace(E_magnitude.min(), E_magnitude.max(), PLOT_PARAMS['3d_contour']['contour_levels'])
    contours = ax.contour(R, Z, E_magnitude, levels=contour_levels, cmap='inferno', linewidths=0.5, linestyles='solid')

    cbar = plt.colorbar(surf, ax=ax, label='Electric field magnitude', shrink=0.6, aspect=20, pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_xlabel('R')
    ax.set_ylabel('Z')
    ax.set_zlabel('Electric Field Magnitude')
    ax.set_title('Electric Field Distribution (3D Contour)')

    ax.view_init(elev=PLOT_PARAMS['3d_contour']['view_angle'][0], azim=PLOT_PARAMS['3d_contour']['view_angle'][1])

    plt.tight_layout()
    save_plot(fig, '3d_contour')

def create_2d_streamplot():
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

    R = np.linspace(*PLOT_PARAMS['2d_streamplot']['R_range'], PLOT_PARAMS['2d_streamplot']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['2d_streamplot']['Z_range'], PLOT_PARAMS['2d_streamplot']['grid_size'])
    R, Z = np.meshgrid(R, Z)

    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    E_magnitude = np.sqrt(E_R_values**2 + E_z_values**2)

    streamplot = ax.streamplot(R, Z, E_R_values, E_z_values, 
                               density=PLOT_PARAMS['2d_streamplot']['density'], 
                               color=E_magnitude, 
                               cmap='viridis', 
                               linewidth=PLOT_PARAMS['2d_streamplot']['linewidth'], 
                               arrowsize=PLOT_PARAMS['2d_streamplot']['arrowsize'])
    
    cbar = plt.colorbar(streamplot.lines, label='Electric field magnitude', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_title('Electric Field Distribution (2D Streamplot)')
    ax.set_xlabel('R')
    ax.set_ylabel('Z')

    plt.tight_layout()
    save_plot(fig, '2d_streamplot')

def create_2d_contour_plot():
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

    R = np.linspace(*PLOT_PARAMS['2d_contour']['R_range'], PLOT_PARAMS['2d_contour']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['2d_contour']['Z_range'], PLOT_PARAMS['2d_contour']['grid_size'])
    R, Z = np.meshgrid(R, Z)

    E_R_values = E_R(R, Z)
    E_z_values = E_z(R, Z)
    E_magnitude = np.sqrt(E_R_values**2 + E_z_values**2)

    contour = ax.contourf(R, Z, E_magnitude, levels=PLOT_PARAMS['2d_contour']['contour_levels'], cmap='viridis')
    cbar = plt.colorbar(contour, label='Electric field magnitude', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_title('Electric Field Distribution (2D Contour)')
    ax.set_xlabel('R')
    ax.set_ylabel('Z')

    plt.tight_layout()
    save_plot(fig, '2d_contour')

# Main execution
def main():
    print_section("Electric Field Plot Generation")
    print(f"{Fore.YELLOW}Starting plot generation...{Style.RESET_ALL}")

    plot_functions = [
        ("3D Quiver", create_3d_quiver_plot),
        ("3D Contour", create_3d_contour_plot),
        ("2D Streamplot", create_2d_streamplot),
        ("2D Contour", create_2d_contour_plot)
    ]

    for plot_type, plot_func in plot_functions:
        create_plot(plot_type, plot_func)

    print_section("Plot Generation Complete")
    print(f"{Fore.GREEN}All plots have been saved in the '{PLOT_CONFIG['output_folder']}' folder.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")
        sys.exit(1)
