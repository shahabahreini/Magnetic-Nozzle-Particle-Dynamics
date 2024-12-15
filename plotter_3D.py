import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors
from collections import defaultdict
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.progress import Progress
from rich import box
from modules import (
    search_for_export_csv,
    extract_parameters_by_file_name,
    list_csv_files,
    list_folders,
    find_common_and_varying_params,  # This is now properly imported
)
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
    "epsphi": r"\epsilon_{\phi}",
    "eps": r"\epsilon",
    "kappa": r"\kappa",
    "deltas": r"\delta_s",
    "beta": r"\beta",
    "alpha": r"\alpha",
}

GREEK_SYMBOLS_InitialCondition = {
    "theta": r"\theta_0",
    "phi": r"\phi_0",
    "alpha": r"\alpha_0",
    "beta": r"\beta_0",
    "time": r"\tau",
    "timestamp": r"\tau",
    "kappa": r"\kappa_0",
    "deltas": r"\delta_s",
    "beta": r"\beta_0",
    "alpha": r"\alpha_0",
    "epsphi": r"\epsilon_{\phi}",
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


def get_axis_label(param, initial_condition=False):
    """Get the appropriate axis label with Greek symbols if applicable."""
    if initial_condition:
        greek_symbol = GREEK_SYMBOLS_InitialCondition.get(param, param)
    else:
        greek_symbol = GREEK_SYMBOLS.get(param, param)
    return f"{greek_symbol}" if greek_symbol != param else param


def plot_2d(ax, df, x_param, y_param, time, color_map, norm, label):
    """Plot 2D scatter and line plots."""
    x_, y_ = df[x_param], df[y_param]
    # Map the colors through the colormap first
    scatter_colors = color_map(norm(time))
    ax.scatter(x_, y_, c=scatter_colors, s=2)
    ax.plot(x_, y_, alpha=0.3, label=label)

    # Set axis labels with Greek symbols if applicable
    ax.set_xlabel(get_axis_label(x_param))
    ax.set_ylabel(get_axis_label(y_param))

def plot_3d(
    ax,
    df,
    coord_system,
    time,
    color_map,
    norm,
    label,
    use_scatter,
    use_time_color,
    show_projections,
):
    """Plot 3D scatter and line plots with projections."""
    if coord_system == "cylindrical":
        x, y, z = df["rho"], df["phi"], df["z"]
        x_label, y_label, z_label = r"$\rho$", r"$\phi$", r"$z$"
    else:
        x, y, z = cylindrical_to_cartesian(df["rho"], df["phi"], df["z"])
        x_label, y_label, z_label = r"$x$", r"$y$", r"$z$"

    # Map the colors through the colormap first
    scatter_colors = color_map(norm(time))

    if use_scatter and use_time_color:
        ax.scatter(x, y, z, c=scatter_colors, s=2)
    elif use_scatter:
        ax.scatter(x, y, z, c="b", s=2)

    if use_time_color:
        # Plot segments with color gradient
        for i in range(len(x)-1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                   c=scatter_colors[i], 
                   alpha=0.3)
    else:
        ax.plot(x, y, z, c="b", alpha=0.3, label=label)

    # Add projections if requested
    if show_projections:
        z_min = np.min(z)
        y_min = np.min(y)
        x_min = np.min(x)
        ax.plot(x, y, [z_min] * len(x), "k--", alpha=0.2)
        ax.plot(x, [y_min] * len(x), z, "k--", alpha=0.2)
        ax.plot([x_min] * len(y), y, z, "k--", alpha=0.2)

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

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


def add_parameter_textbox(ax_params, common_params, varying_params):
    """Add a text box with simulation parameters."""
    ax_params.axis("off")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    param_text = "Simulation Parameters:\n\n"

    # Add common parameters
    for param, value in common_params.items():
        param_text += rf"${get_axis_label(param, True)} = {value}$" + "\n"

    # Add varying parameters
    param_text += "\nVarying Parameters:\n"
    unique_varying_params = set(
        param.split("=")[0] for params in varying_params.values() for param in params
    )
    for param in unique_varying_params:
        param_text += rf"${get_axis_label(param, True)}$" + "\n"

    param_text += f"\nMethod: {CONFIG['method']}"

    ax_params.text(
        0.05,
        0.95,
        param_text,
        transform=ax_params.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )


def Plotter_2d3d(
    file_list,
    folder_path,
    x_param,
    y_param,
    plot_type="2d",
    coord_system="cartesian",
    use_scatter=True,
    use_time_color=True,
    show_projections=False,
    progress=None,
    task=None,
):
    """
    Main plotting function with progress tracking.
    
    Args:
        file_list: List of files to process
        folder_path: Path to the folder containing files
        x_param: Parameter for x-axis
        y_param: Parameter for y-axis
        plot_type: Type of plot ('2d' or '3d')
        coord_system: Coordinate system ('cartesian' or 'cylindrical')
        use_scatter: Whether to use scatter plot
        use_time_color: Whether to use time-based coloring
        show_projections: Whether to show projections (3D only)
        progress: Progress instance for tracking
        task: Task ID for progress tracking
    """
    # Update progress if provided
    if progress and task:
        progress.update(task, description="[cyan]Setting up plot...", advance=10)

    fig, ax, ax_params = setup_plot(plot_type)

    common_params, varying_params, sorted_files = find_common_and_varying_params(
        file_list
    )

    # Update progress
    if progress and task:
        progress.update(task, description="[cyan]Processing files...", advance=10)
        files_per_increment = max(1, len(sorted_files) // 60)  # Divide remaining 60% into steps

    for i, fname in enumerate(sorted_files):
        full_path = os.path.join(folder_path, fname)
        df = pd.read_csv(full_path)
        params = extract_parameters_by_file_name(fname)

        time = df["timestamp"] if "timestamp" in df.columns else df.index
        norm = colors.Normalize(vmin=time.min(), vmax=time.max())
        color_map = plt.cm.viridis

        label = ", ".join(
            [
                f"${GREEK_SYMBOLS.get(param.split('=')[0], param.split('=')[0])}={param.split('=')[1]}$"
                for param in varying_params[fname]
            ]
        )

        if plot_type == "2d":
            plot_2d(ax, df, x_param, y_param, time, color_map, norm, label)
        else:
            x, y, z = plot_3d(
                ax,
                df,
                coord_system,
                time,
                color_map,
                norm,
                label,
                use_scatter,
                use_time_color,
                show_projections,
            )

        # Update progress every few files
        if progress and task and i % files_per_increment == 0:
            progress.update(task, advance=1)

    # Update progress
    if progress and task:
        progress.update(task, description="[cyan]Finalizing plot...", advance=10)

    # Add colorbar only if time coloring is used
    if use_time_color:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=color_map), cax=cbar_ax
        )
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
        ncol=3,
        fontsize=8,
    )

    add_parameter_textbox(ax_params, common_params, varying_params)

    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])

    # Save the plot
    os.makedirs(CONFIG["plots_folder"], exist_ok=True)
    path_to_save = os.path.join(
        CONFIG["plots_folder"], f"multi_plot{CONFIG['save_file_extension']}"
    )
    plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
    console.print(f"[green]Plot saved as:[/green] [bold]{path_to_save}[/bold]")

    # Final progress update
    if progress and task:
        progress.update(task, description="[green]Plot generation complete!", advance=10)

    plt.show()


def validate_parameter(df, param):
    if param not in df.columns:
        console.print(
            f"[red]Error: '{param}' is not a valid parameter. Please choose from the available parameters.[/red]"
        )
        return False
    return True


def get_parameter_from_input(df, prompt_text):
    """Get parameter from user input, accepting either name or number."""
    while True:
        user_input = Prompt.ask(prompt_text)

        # Try to convert input to integer (index)
        try:
            if user_input.isdigit():
                index = int(user_input) - 1
                if 0 <= index < len(df.columns):
                    return df.columns[index]
                else:
                    console.print("[red]Invalid index. Please try again.[/red]")
                    display_available_parameters(df)
                    continue
            # If input is a string (parameter name)
            elif user_input in df.columns:
                return user_input
            else:
                console.print("[red]Invalid parameter name. Please try again.[/red]")
                display_available_parameters(df)
                continue
        except ValueError:
            console.print(
                "[red]Invalid input. Please enter a number or parameter name.[/red]"
            )
            display_available_parameters(df)
            continue


def main():
    console = Console()
    console.print(
        Panel("Welcome to the Particle Trajectory Plotter", style="bold magenta")
    )

    # List CSV files in the folder
    folder_path, selected_files = list_folders()
    print(folder_path, selected_files)

    # Read the first CSV file to get available parameters
    df = pd.read_csv(os.path.join(folder_path, selected_files[0]))

    # Ask user for plot type
    plot_type = Prompt.ask(
        "Enter the plot type (2d for 2D plot, 3d for 3D plot)",
        choices=["2d", "3d"],
        default="3d",
    )

    # For 2D plots, modify the parameter selection part:
    if plot_type == "2d":
        display_available_parameters(df)
        console.print(
            "[yellow]You can enter either the parameter name or its number from the list above[/yellow]"
        )

        x_param = get_parameter_from_input(
            df, "Enter the parameter for x-axis (name or number)"
        )
        y_param = get_parameter_from_input(
            df, "Enter the parameter for y-axis (name or number)"
        )
        coord_system = "cartesian"
    else:
        coord_system = Prompt.ask(
            "Enter the coordinate system (cartesian or cylindrical)",
            choices=["cartesian", "cylindrical"],
            default="cartesian",
        )
        x_param = y_param = None

    # Ask for additional plot options
    use_scatter = Confirm.ask("Use scatter plot? (y/n)", default=False)
    use_time_color = Confirm.ask("Use time-based coloring? (y/n)", default=False)
    show_projections = (
        Confirm.ask("Show projections? (3D only) (y/n)", default=False)
        if plot_type == "3d"
        else False
    )

    # Generate the plot
    try:
        fig, ax = Plotter_2d3d(
            selected_files,
            folder_path,
            x_param,
            y_param,
            plot_type,
            coord_system,
            use_scatter,
            use_time_color,
            show_projections,
        )

        if fig is None or ax is None:
            console.print(
                "[red]Error: Plotter function returned None. Please check your input parameters.[/red]"
            )
            return

        # Save the plot
        save_path = os.path.join(folder_path, CONFIG["plots_folder"])
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(
            save_path, f"{CONFIG['save_file_name']}{CONFIG['save_file_extension']}"
        )
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        console.print(f"[green]Plot saved as {save_file}[/green]")

        # Show the plot
        plt.show()
    except Exception as e:
        console.print(f"[red]An error occurred: {str(e)}[/red]")


if __name__ == "__main__":
    main()
