import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress
from lib import search_for_export_csv, extract_parameters_by_file_name, list_csv_files, list_folders
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


console = Console()

# Configuration
CONFIG = {
    "save_file_name": "R_Z",
    "save_file_extension": ".png",
    "do_plot_line_from_origin": False,
    "is_multi_files": False,
    "target_folder_multi_files": "/csv/nomagnetic2/",
    "plots_folder": "plots",
    "simulation_time": 100,
    "method": "Feagin14",
}

# Default parameter dictionary
DEFAULT_PARAMS = {
    "eps": 0.0177,
    "epsphi": 0.0,
    "kappa": 1.0,
    "deltas": 1.0,
    "beta": 90.0,
    "alpha": 30.0,
    "theta": 30.0,
    "time": 100,
}

# Greek symbol mapping for axis labels
GREEK_SYMBOLS = {
    "x": r"x",
    "y": r"y",
    "z": r"z",
    "vx": r"v_x",
    "vy": r"v_y",
    "vz": r"v_z",
    "theta": r"\theta",
    "phi": r"\phi",
    "alpha": r"\alpha",
    "beta": r"\beta",
    "time": r"\tau",
    "timestamp": r"\tau",
    "dphi": r"v_\phi",
    "dtheta": r"v_\theta",
    "dalpha": r"v_\alpha",
    "dbeta": r"v_\beta",
    "rho": r"\rho",
    "drho": r"v_\rho",
}


def cylindrical_to_cartesian(rho, phi, z):
    """Convert cylindrical coordinates to Cartesian coordinates."""
    rho, phi, z = map(np.array, (rho, phi, z))
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z


def display_available_parameters(df):
    """Display available parameters for plotting in a rich table."""
    table = Table(title="Available Parameters for Plotting")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Parameter", style="magenta")
    for i, column in enumerate(df.columns, 1):
        table.add_row(str(i), column)
    console.print(table)


def setup_plot(plot_type):
    """Set up the plot based on the plot type."""
    if plot_type == "2d":
        fig, (ax, ax_params) = plt.subplots(
            1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [3, 1]}
        )
    else:
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0], projection="3d")
        ax_params = fig.add_subplot(gs[1])
    return fig, ax, ax_params


def plot_2d(ax, df, x_param, y_param, time, color_map, norm, kappa):
    """Plot 2D scatter and line plots."""
    x_, y_ = df[x_param], df[y_param]
    ax.scatter(x_, y_, c=time, cmap=color_map, norm=norm, s=2)
    ax.plot(x_, y_, alpha=0.3, label=rf"$\kappa = {kappa}$")


def plot_3d(ax, df, coord_system, time, color_map, norm, kappa):
    """Plot 3D scatter and line plots with projections."""
    if coord_system == "cylindrical":
        x, y, z = df["rho"], df["phi"], df["z"]
    else:
        x, y, z = cylindrical_to_cartesian(df["rho"], df["phi"], df["z"])

    ax.scatter(x, y, z, c=time, cmap=color_map, norm=norm, s=2)
    ax.plot(x, y, z, alpha=0.3, label=rf"$\kappa = {kappa}$")

    # Add projections
    ax.plot(x, y, min(z), "k--", alpha=0.2)
    ax.plot(x, [min(y)] * len(x), z, "k--", alpha=0.2)
    ax.plot([min(x)] * len(y), y, z, "k--", alpha=0.2)
    return x, y, z


def set_axis_properties(ax, plot_type, x_param, y_param):
    """Set axis properties based on plot type."""
    if plot_type == "2d":
        x_label = GREEK_SYMBOLS.get(x_param, x_param)
        y_label = GREEK_SYMBOLS.get(y_param, y_param)
        ax.set_xlabel(f"${x_label}$", labelpad=10)
        ax.set_ylabel(f"${y_label}$", labelpad=10)
        ax.set_title(f"${y_label}$ vs ${x_label}$", pad=20)
    else:
        ax.set_xlabel("$x$", labelpad=10)
        ax.set_ylabel("$y$", labelpad=10)
        ax.set_zlabel("$z$", labelpad=10)
        ax.set_title("3D Trajectory (Equal Scale)", pad=20)
        ax.view_init(elev=20, azim=45)
        ax.grid(True, linestyle="--", alpha=0.5)


def set_3d_equal_aspect(ax, x, y, z):
    """Set equal aspect ratio for 3D plot."""
    max_range = (
        np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
    )
    mid_x, mid_y, mid_z = [(x.max() + x.min()) * 0.5 for x in (x, y, z)]
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def add_parameter_textbox(ax_params, params):
    """Add a text box with simulation parameters."""
    ax_params.axis("off")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    param_text = (
        "Simulation Parameters:\n\n"
        f"$\\theta_0$ = {params['theta']}°\n"
        f"$\\alpha_0$ = {params['alpha']}°\n"
        f"$\\beta_0$ = {params['beta']}°\n"
        f"$\\phi_0$ = 0.0°\n"
        f"$\\delta^*$ = {params['deltas']}\n"
        f"$\\varepsilon_\\phi$ = {params['epsphi']}\n"
        f"$\\varepsilon$ = {params['eps']}\n"
        f"$\\kappa$ = {params['kappa']}\n"
        f"$\\tau$ = {params['time']}\n\n"
        f"Method: {CONFIG['method']}"
    )
    ax_params.text(
        0.05,
        0.95,
        param_text,
        transform=ax_params.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )


def Plotter(file_list, folder_path, x_param, y_param, plot_type="2d", coord_system="cartesian"):
    """Main plotting function."""
    fig, ax, ax_params = setup_plot(plot_type)

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(file_list))

        for fname in file_list:
            full_path = os.path.join(folder_path, fname)
            df = pd.read_csv(full_path)
            params = extract_parameters_by_file_name(fname)

            time = df["timestamp"] if "timestamp" in df.columns else df.index
            norm = colors.Normalize(vmin=time.min(), vmax=time.max())
            color_map = plt.cm.viridis

            if plot_type == "2d":
                plot_2d(
                    ax, df, x_param, y_param, time, color_map, norm, params["kappa"]
                )
            else:
                x, y, z = plot_3d(
                    ax, df, coord_system, time, color_map, norm, params["kappa"]
                )

            progress.update(task, advance=1)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=color_map), cax=cbar_ax)
    cbar.set_label("Time", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)

    set_axis_properties(ax, plot_type, x_param, y_param)
    ax.tick_params(axis="both", which="major", labelsize=8)

    if plot_type == "3d":
        set_3d_equal_aspect(ax, x, y, z)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15 if plot_type == "2d" else -0.05),
        fancybox=True,
        shadow=True,
        ncol=5,
        fontsize=8,
    )

    add_parameter_textbox(ax_params, params)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Save the plot
    os.makedirs(CONFIG["plots_folder"], exist_ok=True)
    path_to_save = os.path.join(
        CONFIG["plots_folder"], f"multi_plot{CONFIG['save_file_extension']}"
    )
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    console.print(f"[green]Plot saved as:[/green] [bold]{path_to_save}[/bold]")
    plt.show()

def main():
    """Main function to run the interactive plotter."""
    console.print(
        Panel.fit("[bold cyan]Welcome to the Interactive Plotter[/bold cyan]")
    )

    # Ask the user for the mode: folder selection or single file from current directory
    console.print("[bold]Select mode of operation:[/bold]")
    console.print("1. Multi-file mode")
    console.print("2. Single-file mode")
    mode_choice = IntPrompt.ask("Enter your choice", choices=["1", "2"])

    # Set the configuration based on the mode choice
    CONFIG['is_multi_files'] = (mode_choice == 1)

    if CONFIG["is_multi_files"]:
        # Folder selection mode
        console.print("[bold]Select a folder containing CSV files:[/bold]")
        folders = list_folders()
        folder_choice = IntPrompt.ask("Enter a folder number", choices=[str(i) for i in range(1, len(folders) + 1)])
        selected_folder = folders[folder_choice - 1]

        folder_path = os.path.join('.', selected_folder)
        files = list_csv_files(folder_path)

        console.print(
            "\n[bold]Select files to plot (enter numbers separated by space or press Enter to select all):[/bold]")
        file_choice = Prompt.ask("Enter file numbers or press Enter")
        if file_choice:
            try:
                chosen_indices = [int(i) - 1 for i in file_choice.split()]
                selected_files = [files[i] for i in chosen_indices if 0 <= i < len(files)]
                if not selected_files:
                    raise ValueError("Invalid file numbers")
            except ValueError:
                console.print("[red]Invalid input! Please enter valid file numbers separated by space.[/red]")
                return
        else:
            selected_files = files  # Plot all files if Enter is pressed without input
        
        console.print("\n[yellow]Listing CSV files in the target folder...[/yellow]")

    else:
        # Single file selection from the current directory
        folder_path = os.getcwd()  # Current directory
        files = list_csv_files(folder_path)
        file_choice = IntPrompt.ask(
            "Choose a file (enter a number from the list)",
            choices=[str(i) for i in range(1, len(files) + 1)],
        )
        selected_files = [files[file_choice - 1]]

    console.print("[yellow]Reading CSV file(s)...[/yellow]")
    df = pd.read_csv(os.path.join(folder_path, selected_files[0]))
    console.print("[green]CSV file(s) loaded successfully![/green]")

    plot_type = Prompt.ask("Enter plot type", choices=["2d", "3d"], default="2d")

    if plot_type == "2d":
        console.print("\n[bold]Available parameters for plotting:[/bold]")
        display_available_parameters(df)

        x_index = int(
            Prompt.ask("Enter the index number for x-axis parameter", default="1")
        )
        y_index = int(
            Prompt.ask("Enter the index number for y-axis parameter", default="2")
        )

        x_param = df.columns[x_index - 1]
        y_param = df.columns[y_index - 1]

        console.print(f"[green]Selected x-axis:[/green] [bold]{x_param}[/bold]")
        console.print(f"[green]Selected y-axis:[/green] [bold]{y_param}[/bold]")
        coord_system = None
    else:
        coord_system = Prompt.ask(
            "Enter coordinate system",
            choices=["cartesian", "cylindrical"],
            default="cartesian",
        )
        x_param = y_param = None

    Plotter(selected_files, folder_path, x_param, y_param, plot_type, coord_system)

if __name__ == "__main__":
    main()
