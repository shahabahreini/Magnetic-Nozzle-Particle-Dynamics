import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import os
import re
import numpy as np
from lib import search_for_export_csv, extract_parameters_by_file_name
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from matplotlib import gridspec


console = Console()

# ---------------------------------- Config ---------------------------------- #
save_file_name = r"R_Z"
save_file_extension = ".png"

do_plot_line_from_origin = False

is_multi_files = False
target_folder_multi_files = "/csv/nomagnetic2/"
plots_folder = "plots"

# doesn't need to update the parameters if individual file is used
parameter_dict = {
    "eps": 0.0177,
    "epsphi": 0.0,
    "kappa": 1.0,
    "deltas": 1.0,
    "beta": 90.0,
    "alpha": 30.0,
    "theta": 30.0,
    "time": 100,
}

simulation_time = 100
method = "Feagin14"
# ------------------------------------ --- ----------------------------------- #

def tangent_calculator(x_, y_):
    return y_ / x_

def display_available_parameters(df):
    table = Table(title="Available Parameters for Plotting")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Parameter", style="magenta")

    for i, column in enumerate(df.columns, 1):
        table.add_row(str(i), column)

    console.print(table)

def point_finder(x_i, y_i, x_f):
    b = 0
    m = tangent_calculator(x_i, y_i)
    y_f = x_f * m + b

    xs = [x_i, x_f]
    ys = [y_i, y_f]
    return xs, ys

def cylindrical_to_cartesian(rho, phi, z):
    rho = rho.to_numpy()
    phi = phi.to_numpy()
    z = z.to_numpy()
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    return x, y, z

def Plotter(save_filename, x_param, y_param, plot_type='2d', coord_system='cartesian'):
    global parameter_dict

    if is_multi_files:
        path_ = target_folder_multi_files
        filelst = os.listdir(path_)
        filelst = [f for f in filelst if f.endswith('.csv')]
    else:
        path_ = ""
        filelst = [save_filename]
        
    if not filelst:
        console.print("[bold red]No CSV files found.[/bold red]")
        return

    parameter_dict = extract_parameters_by_file_name(filelst[0])

    eps = parameter_dict["eps"]
    epsphi = parameter_dict["epsphi"]
    kappa = parameter_dict["kappa"]
    deltas = parameter_dict["deltas"]
    beta = parameter_dict["beta"]
    alpha = parameter_dict["alpha"]
    theta = parameter_dict["theta"]
    simulation_time = parameter_dict["time"]
    
    if plot_type == '2d':
        fig, (ax, ax_params) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})
    else:
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0], projection='3d')
        ax_params = fig.add_subplot(gs[1])
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(filelst))
        
        for fname in filelst:
            full_path = os.path.join(path_, fname)
            data = pd.read_csv(full_path)
            df = pd.DataFrame(data)

            parameter_dict = extract_parameters_by_file_name(fname)
            kappa = parameter_dict.get('kappa', 'Unknown')

            # Create a color map based on time
            time = df['timestamp'] if 'timestamp' in df.columns else df.index
            norm = colors.Normalize(vmin=time.min(), vmax=time.max())
            color_map = plt.cm.viridis

            if plot_type == '2d':
                x_ = df[x_param]
                y_ = df[y_param]
                points = ax.scatter(x_, y_, c=time, cmap=color_map, norm=norm, s=2)
                ax.plot(x_, y_, alpha=0.3, label=fr"$\kappa = {kappa}$")
            else:  # 3d plot
                if coord_system == 'cylindrical':
                    rho, phi, z = df['rho'], df['phi'], df['z']
                    x, y, z = cylindrical_to_cartesian(rho, phi, z)
                else:  # cartesian
                    if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                        x, y, z = df['x'], df['y'], df['z']
                    else:
                        rho, phi, z = df['rho'], df['phi'], df['z']
                        x, y, z = cylindrical_to_cartesian(rho, phi, z)
                
                points = ax.scatter(x, y, z, c=time, cmap=color_map, norm=norm, s=2)
                ax.plot(x, y, z, alpha=0.3, label=fr"$\kappa = {kappa}$")
                
                # Add projections
                ax.plot(x, y, min(z), 'k--', alpha=0.2)  # x-y projection
                ax.plot(x, [min(y)]*len(x), z, 'k--', alpha=0.2)  # x-z projection
                ax.plot([min(x)]*len(y), y, z, 'k--', alpha=0.2)  # y-z projection

            progress.update(task, advance=1)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(points, cax=cbar_ax)
    cbar.set_label('Time', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)# Adjust tick label size

    # Update x and y labels with Greek symbols and velocity representations
    greek_symbols = {
        'x': r'x', 'y': r'y', 'z': r'z',
        'vx': r'v_x', 'vy': r'v_y', 'vz': r'v_z',
        'theta': r'\theta', 'phi': r'\phi',
        'alpha': r'\alpha', 'beta': r'\beta',
        'time': r'\tau',
        'timestamp': r'\tau',
        'dphi': r'v_\phi',
        'dtheta': r'v_\theta',
        'dalpha': r'v_\alpha',
        'dbeta': r'v_\beta',
        'rho': r'\rho',
        'drho': r'v_\rho'
    }

    if plot_type == '2d':
        x_label = greek_symbols.get(x_param, x_param)
        y_label = greek_symbols.get(y_param, y_param)
        ax.set_xlabel(f'${x_label}$', labelpad=10)
        ax.set_ylabel(f'${y_label}$', labelpad=10)
        ax.set_title(f'${y_label}$ vs ${x_label}$', pad=20)
    else:
        ax.set_xlabel('$x$', labelpad=10)
        ax.set_ylabel('$y$', labelpad=10)
        ax.set_zlabel('$z$', labelpad=10)
        ax.set_title("3D Trajectory (Equal Scale)", pad=20)

        # Adjust viewing angle
        ax.view_init(elev=20, azim=45)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Adjust tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Adjust legend
    if plot_type == '3d':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5, fontsize=8)
    else:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  fancybox=True, shadow=True, ncol=5, fontsize=8)

    # Create a text box for simulation parameters
    ax_params.axis('off')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    param_text = (
        "Simulation Parameters:\n\n"
        f"$\\theta_0$ = {theta}°\n"
        f"$\\alpha_0$ = {alpha}°\n"
        f"$\\beta_0$ = {beta}°\n"
        f"$\\phi_0$ = 0.0°\n"
        f"$\\delta^*$ = {deltas}\n"
        f"$\\varepsilon_\\phi$ = {epsphi}\n"
        f"$\\varepsilon$ = {eps}\n"
        f"$\\kappa$ = {kappa}\n"
        f"$\\tau$ = {simulation_time}\n\n"
        f"Method: {method}"
    )
    ax_params.text(0.05, 0.95, param_text, transform=ax_params.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    path_to_save = os.path.join(plots_folder, str(save_filename + save_file_extension))
    plt.savefig(path_to_save, dpi=300, bbox_inches='tight')
    console.print(f"[green]Plot saved as:[/green] [bold]{path_to_save}[/bold]")
    plt.show()

if __name__ == "__main__":
    console.print(Panel.fit("[bold cyan]Welcome to the Interactive Plotter[/bold cyan]"))

    if not is_multi_files:
        chosen_csv = search_for_export_csv()
        individual_file_names_to_read = [chosen_csv]
    else:
        chosen_csv = "multi_plot"

    console.print("[yellow]Reading CSV file...[/yellow]")
    data = pd.read_csv(individual_file_names_to_read[0])
    df = pd.DataFrame(data)
    console.print("[green]CSV file loaded successfully![/green]")

    plot_type = Prompt.ask("Enter plot type", choices=["2d", "3d"], default="2d")

    if plot_type == '2d':
        console.print("\n[bold]Available parameters for plotting:[/bold]")
        display_available_parameters(df)
        
        x_index = int(Prompt.ask("Enter the index number for x-axis parameter", default="1"))
        y_index = int(Prompt.ask("Enter the index number for y-axis parameter", default="2"))
        
        x_param = df.columns[x_index - 1]
        y_param = df.columns[y_index - 1]
        
        console.print(f"[green]Selected x-axis:[/green] [bold]{x_param}[/bold]")
        console.print(f"[green]Selected y-axis:[/green] [bold]{y_param}[/bold]")
        coord_system = None
    else:  # 3d plot
        coord_system = Prompt.ask("Enter coordinate system", choices=["cartesian", "cylindrical"], default="cartesian")
        x_param = y_param = None  # These are not used for 3D plots

    Plotter(chosen_csv, x_param, y_param, plot_type, coord_system)

