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
    'output_folder': os.path.join('plots', 'BField_Plots'),  # Folder to save plots
    'show_plots': False,             # Set to True to display plots on screen after saving
}

# Physical constants
PHYSICS_PARAMS = {
    'B0': 1.0,      # Magnetic field strength at the magnetic axis (Tesla)
    'r0': 1.0,      # Reference radius (meter)
}

# Plot-specific parameters
PLOT_PARAMS = {
    '2d_streamplot': {
        'grid_size': 100,        # Number of grid points in each dimension
        'R_range': (-10, 10),    # Range of R values (min, max)
        'Z_range': (-10, 10),    # Range of Z values (min, max)
        'density': 2,            # Density of streamlines
        'linewidth': 1,          # Width of streamlines
        'arrowsize': 1.5,        # Size of arrows on streamlines
    },
    '3d_quiver': {
        'grid_size': 8,          # Number of grid points in each dimension
        'R_range': (-2, 2),      # Range of R values (min, max)
        'Z_range': (-2, 2),      # Range of Z values (min, max)
        'Y_range': (-2, 2),      # Range of Y values (min, max)
        'quiver_length': 0.5,    # Length of quiver arrows
    },
    '2d_contour': {
        'grid_size': 100,        # Number of grid points in each dimension
        'R_range': (-2, 2),      # Range of R values (min, max)
        'Z_range': (-2, 2),      # Range of Z values (min, max)
        'contour_levels': 20,    # Number of contour levels
    },
    '3d_contour': {
        'grid_size': 10,         # Number of grid points in each dimension
        'R_range': (-2, 2),      # Range of R values (min, max)
        'Z_range': (-2, 2),      # Range of Z values (min, max)
        'Y_range': (-2, 2),      # Range of Y values (min, max)
        'contour_levels': 400,   # Number of contour levels
    },
    '2d_quiver': {
        'grid_size': 15,         # Number of grid points in each dimension
        'R_range': (-2, 2),      # Range of R values (min, max)
        'Z_range': (-2, 2),      # Range of Z values (min, max)
        'scale': 80,             # Scale for quiver plot
        'headwidth': 4,          # Width of arrow head
        'headlength': 4,         # Length of arrow head
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
def calculate_B_field(R, Z):
    r = np.sqrt(R**2 + Z**2)
    BR = PHYSICS_PARAMS['B0'] * PHYSICS_PARAMS['r0']**2 * R / (R**2 + Z**2)**(3/2)
    BZ = PHYSICS_PARAMS['B0'] * PHYSICS_PARAMS['r0']**2 * Z / (R**2 + Z**2)**(3/2)
    return BR, BZ

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
    filename = f"magnetic_field_{plot_type}_{timestamp}.png"
    filepath = os.path.join(PLOT_CONFIG['output_folder'], filename)
    fig.savefig(filepath, bbox_inches='tight')
    print(f"{Fore.GREEN}✔ Plot saved: {filepath}{Style.RESET_ALL}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close(fig)

def create_2d_streamplot():
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

    R = np.linspace(*PLOT_PARAMS['2d_streamplot']['R_range'], PLOT_PARAMS['2d_streamplot']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['2d_streamplot']['Z_range'], PLOT_PARAMS['2d_streamplot']['grid_size'])
    R, Z = np.meshgrid(R, Z)

    BR, BZ = calculate_B_field(R, Z)
    B_mag = np.sqrt(BR**2 + BZ**2)

    streamplot = ax.streamplot(R, Z, BR, BZ, 
                               density=PLOT_PARAMS['2d_streamplot']['density'], 
                               color=B_mag, 
                               cmap='viridis', 
                               linewidth=PLOT_PARAMS['2d_streamplot']['linewidth'], 
                               arrowsize=PLOT_PARAMS['2d_streamplot']['arrowsize'])
    
    cbar = plt.colorbar(streamplot.lines, label='Magnetic Field Strength (T)', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_title('2D Magnetic Field Streamlines')
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')

    plt.tight_layout()
    save_plot(fig, '2d_streamplot')

def create_3d_quiver_plot():
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size'])
    ax = fig.add_subplot(111, projection='3d')

    R = np.linspace(*PLOT_PARAMS['3d_quiver']['R_range'], PLOT_PARAMS['3d_quiver']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['3d_quiver']['Z_range'], PLOT_PARAMS['3d_quiver']['grid_size'])
    Y = np.linspace(*PLOT_PARAMS['3d_quiver']['Y_range'], PLOT_PARAMS['3d_quiver']['grid_size'])
    R, Y, Z = np.meshgrid(R, Y, Z)

    BR, BZ = calculate_B_field(np.sqrt(R**2 + Y**2), Z)
    BX = BR * R / np.sqrt(R**2 + Y**2)
    BY = BR * Y / np.sqrt(R**2 + Y**2)
    B_mag = np.sqrt(BX**2 + BY**2 + BZ**2)

    norm = plt.Normalize(vmin=0, vmax=B_mag.max())
    colors = plt.cm.viridis(norm(B_mag))

    quiver = ax.quiver(R, Y, Z, BX, BY, BZ, length=PLOT_PARAMS['3d_quiver']['quiver_length'], 
                       normalize=True, color=colors.reshape(-1, 4))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Magnetic Field Strength (T)', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Magnetic Field Vector Field')

    plt.tight_layout()
    save_plot(fig, '3d_quiver')

def create_2d_contour_plot():
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

    R = np.linspace(*PLOT_PARAMS['2d_contour']['R_range'], PLOT_PARAMS['2d_contour']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['2d_contour']['Z_range'], PLOT_PARAMS['2d_contour']['grid_size'])
    R, Z = np.meshgrid(R, Z)

    BR, BZ = calculate_B_field(R, Z)
    B_mag = np.sqrt(BR**2 + BZ**2)

    contour = ax.contourf(R, Z, B_mag, levels=PLOT_PARAMS['2d_contour']['contour_levels'], cmap='viridis')
    cbar = plt.colorbar(contour, label='Magnetic Field Strength (T)', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_title('2D Magnetic Field Contour Plot')
    ax.set_xlabel('R (m)')
    ax.set_ylabel('Z (m)')

    plt.tight_layout()
    save_plot(fig, '2d_contour')

def create_3d_contour_plot():
    fig = plt.figure(figsize=PLOT_CONFIG['figure_size'])
    ax = fig.add_subplot(111, projection='3d')

    R = np.linspace(*PLOT_PARAMS['3d_contour']['R_range'], PLOT_PARAMS['3d_contour']['grid_size'])
    Z = np.linspace(*PLOT_PARAMS['3d_contour']['Z_range'], PLOT_PARAMS['3d_contour']['grid_size'])
    Y = np.linspace(*PLOT_PARAMS['3d_contour']['Y_range'], PLOT_PARAMS['3d_contour']['grid_size'])
    R, Y, Z = np.meshgrid(R, Y, Z)

    BR, BZ = calculate_B_field(np.sqrt(R**2 + Y**2), Z)
    BX = BR * R / np.sqrt(R**2 + Y**2)
    BY = BR * Y / np.sqrt(R**2 + Y**2)
    B_mag = np.sqrt(BX**2 + BY**2 + BZ**2)

    contour = ax.contourf(R[:,:,0], Y[:,:,0], B_mag[:,:,PLOT_PARAMS['3d_contour']['grid_size']//2], 
                          levels=PLOT_PARAMS['3d_contour']['contour_levels'], cmap='viridis', zdir='z', 
                          offset=PLOT_PARAMS['3d_contour']['Z_range'][0])
    
    contour = ax.contourf(R[:,0,:], B_mag[:,PLOT_PARAMS['3d_contour']['grid_size']//2,:], Z[:,0,:], 
                          levels=PLOT_PARAMS['3d_contour']['contour_levels'], cmap='viridis', zdir='y', 
                          offset=PLOT_PARAMS['3d_contour']['Y_range'][1])
    
    contour = ax.contourf(B_mag[PLOT_PARAMS['3d_contour']['grid_size']//2,:,:], Y[0,:,:], Z[0,:,:], 
                          levels=PLOT_PARAMS['3d_contour']['contour_levels'], cmap='viridis', zdir='x', 
                          offset=PLOT_PARAMS['3d_contour']['R_range'][0])

    cbar = fig.colorbar(contour, label='Magnetic Field Strength (T)', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_title('3D Magnetic Field Contour Plot')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    plt.tight_layout()
    save_plot(fig, '3d_contour')

def create_2d_quiver_plot():
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])

    # Fetch parameters from config
    grid_size = PLOT_PARAMS['2d_quiver']['grid_size']
    R_range = PLOT_PARAMS['2d_quiver']['R_range']
    Z_range = PLOT_PARAMS['2d_quiver']['Z_range']
    scale = PLOT_PARAMS['2d_quiver']['scale']
    headwidth = PLOT_PARAMS['2d_quiver']['headwidth']
    headlength = PLOT_PARAMS['2d_quiver']['headlength']

    R = np.linspace(*R_range, grid_size)
    Z = np.linspace(*Z_range, grid_size)
    R, Z = np.meshgrid(R, Z)

    BR, BZ = calculate_B_field(R, Z)
    B_mag = np.sqrt(BR**2 + BZ**2)

    quiver = ax.quiver(R, Z, BR, BZ, B_mag, cmap='viridis', pivot='mid', scale=scale, headwidth=headwidth, headlength=headlength)
    cbar = plt.colorbar(quiver, label='Magnetic Field Strength (T)', pad=PLOT_CONFIG['colorbar_pad'])
    cbar.ax.tick_params(labelsize=PLOT_CONFIG['tick_label_size'])

    ax.set_title('2D Magnetic Field Quiver Plot', fontsize=PLOT_CONFIG['font_size'] + 2)
    ax.set_xlabel('R (m)', fontsize=PLOT_CONFIG['font_size'])
    ax.set_ylabel('Z (m)', fontsize=PLOT_CONFIG['font_size'])

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    save_plot(fig, '2d_quiver')

# Main execution
if __name__ == "__main__":
    print_section("Magnetic Field Plotter")
    
    create_plot("2D Streamplot", create_2d_streamplot)
    create_plot("3D Quiver Plot", create_3d_quiver_plot)
    create_plot("2D Contour Plot", create_2d_contour_plot)
    create_plot("3D Contour Plot", create_3d_contour_plot)
    create_plot("2D Quiver Plot", create_2d_quiver_plot)

    print(f"\n{Fore.GREEN}All plots have been generated and saved in: {PLOT_CONFIG['output_folder']}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}Configuration used:")
    print(f"B0 = {PHYSICS_PARAMS['B0']} T")
    print(f"r0 = {PHYSICS_PARAMS['r0']} m{Style.RESET_ALL}")