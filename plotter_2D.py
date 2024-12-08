import os
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.text import Text
from rich import box
import matplotlib.pyplot as plt
import re
from datetime import datetime
from collections import defaultdict

# Replace the old lib import with specific imports from particle_sim
from modules import (
    get_axis_label,  # from visualization.py
    list_csv_files,  # from file_utils.py
    list_folders,  # from file_utils.py
    find_common_and_varying_params,
    Configuration,
)

console = Console()
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = Configuration(config_path)


def create_parameter_selector(parameters, selected_params=None):
    """
    Create an interactive parameter selector with a better UI.

    Args:
        parameters (list): List of available parameters
        selected_params (list, optional): List of already selected parameters to exclude

    Returns:
        tuple: (selected_parameter, selected_parameter_name)
    """
    console = Console()

    # Create a table for parameter display
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Choice", style="cyan", justify="center")
    table.add_column("Parameter", style="green")
    table.add_column("Description", style="yellow")

    # Parameter descriptions (you can expand this dictionary)
    param_descriptions = {
        "timestamp": "Time value for the simulation",
        "drho": "Change in radial coordinate",
        "dz": "Change in axial coordinate",
        "rho": "Radial coordinate",
        "z": "Axial coordinate",
        "omega_rho": "Radial angular velocity",
        "omega_z": "Axial angular velocity",
    }

    # Filter out already selected parameters
    available_params = [
        p for p in parameters if selected_params is None or p not in selected_params
    ]

    # Add rows to the table
    for i, param in enumerate(available_params, 1):
        description = param_descriptions.get(param, "No description available")
        table.add_row(str(i), param, description)

    # Print the layout
    console.print(table)

    # Get user input with validation
    while True:
        try:
            choice = IntPrompt.ask(
                "Enter your choice",
                choices=[str(i) for i in range(1, len(available_params) + 1)],
            )
            selected_param = available_params[choice - 1]
            return choice, selected_param
        except (ValueError, IndexError):
            console.print("[red]Invalid choice. Please try again.[/red]")


def select_plotting_parameters(parameters):
    """
    Main function to handle the parameter selection process.

    Args:
        parameters (list): List of available parameters

    Returns:
        tuple: (x_param, y_param)
    """
    console = Console()
    selected_params = []
    axis_names = ["x-axis", "y-axis"]
    final_selections = {}

    for axis in axis_names:
        console.print(f"\n[bold]Selecting parameter for {axis}:[/bold]")
        _, selected_param = create_parameter_selector(parameters, selected_params)
        selected_params.append(selected_param)
        final_selections[axis] = selected_param

        # Show confirmation
        console.print(f"\n[green]Selected {axis} parameter: {selected_param}[/green]")

    # Show final selection summary
    console.print("\n[bold]Final Parameter Selection:[/bold]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Axis", style="cyan")
    summary_table.add_column("Parameter", style="green")

    for axis, param in final_selections.items():
        summary_table.add_row(axis, param)

    console.print(summary_table)

    return final_selections["x-axis"], final_selections["y-axis"]


def plot_csv_files(files, folder, x_param, y_param, mode, progress, task):
    console.print(
        f"\n[green]Generating plot: {get_axis_label(y_param)} vs {get_axis_label(x_param)}[/green]"
    )

    # Initialize common_params and varying_params before the loop
    common_params = {}
    varying_params = {}

    # Determine if we're in multi-file mode
    is_multi_files = len(files) > 1

    # If in multi-file mode, get common and varying parameters first
    if is_multi_files:
        common_params, varying_params, sorted_files = find_common_and_varying_params(
            files
        )
        files_to_process = sorted_files
    else:
        files_to_process = files

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Process each file
    for file in files_to_process:
        # Read the CSV file
        df = pd.read_csv(os.path.join(folder, file))

        # Validate columns
        missing_columns = []
        for param in [x_param, y_param]:
            if param not in df.columns:
                missing_columns.append(param)

        if missing_columns:
            console.print(
                f"[red]Error: The following columns are missing in the CSV file {file}:[/red]"
            )
            console.print(f"[red]{', '.join(missing_columns)}[/red]")
            console.print(
                f"[yellow]Available columns: {', '.join(df.columns)}[/yellow]"
            )
            return

        # Plot the data
        if is_multi_files:
            # Use varying parameters for legend in multi-file mode
            varying_param_str = ", ".join(varying_params[file])
            plt.plot(df[x_param], df[y_param], label=varying_param_str)
        else:
            # Simple plot for single file mode
            plt.plot(df[x_param], df[y_param])

    # Update progress bar (40% for generating the plot)
    progress.update(task, advance=40)

    # Set plot title and labels
    plt.suptitle(f"{get_axis_label(y_param)} vs {get_axis_label(x_param)}", fontsize=12)

    # Add common parameters to title if in multi-file mode
    if is_multi_files and common_params:
        common_param_str = ", ".join(
            [f"{get_axis_label(k)}={v}" for k, v in common_params.items()]
        )
        plt.title(
            common_param_str, loc="right", fontsize=8, color="grey", style="italic"
        )

    plt.xlabel(get_axis_label(x_param))
    plt.ylabel(get_axis_label(y_param))

    # Show legend only in multi-file mode
    if is_multi_files:
        plt.legend()

    plt.tight_layout()

    # Update progress bar (20% for finalizing plot layout)
    progress.update(task, advance=20)

    # Generate file name based on mode and timestamp
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    plot_filename = f"{mode}-{current_time}.png"

    # Create plots folder if it doesn't exist
    plots_folder = os.path.join(".", "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Save plot to the plots folder
    plot_path = os.path.join(plots_folder, plot_filename)
    plt.savefig(plot_path, dpi=600)

    # Update progress bar (20% for saving the plot)
    progress.update(task, advance=20)

    console.print(f"\n[green]Plot saved as {plot_path}[/green]")
    plt.show()

    # Update progress bar (final 10% for displaying the plot)
    progress.update(task, advance=10)


def main():
    # Ask the user for the mode: folder selection or single file from current directory
    console.print("[bold]Select mode of operation:[/bold]")
    console.print("1. Select files from a folder")
    console.print("2. Select a single file from the current directory")
    mode_choice = IntPrompt.ask("Enter your choice", choices=["1", "2"])

    if mode_choice == 1:
        # Folder selection mode
        console.print("[bold]Select a folder containing CSV files:[/bold]")
        selected_folder, selected_files = list_folders()
        folder_path = os.path.join(".", selected_folder)
        mode = "multimode"

    else:
        # Single file selection from the current directory
        folder_path = os.getcwd()  # Current directory
        selected_file, all_files = list_csv_files(folder_path)
        selected_files = [selected_file]
        print(selected_files)
        mode = "singlemode"

    # Parameter options
    param_options = ["timestamp", "drho", "dz", "rho", "z", "omega_rho", "omega_z"]

    try:
        x_param, y_param = select_plotting_parameters(param_options)
        print(x_param, y_param)
        print(f"\nFinal selection - X: {x_param}, Y: {y_param}")
    except KeyboardInterrupt:
        print("\nSelection cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

    # Progress bar for plotting (representing file reading, plot generation, saving, and displaying)
    with Progress() as progress:
        task = progress.add_task("[cyan]Plotting...", total=100)
        plot_csv_files(
            selected_files,
            folder_path,
            x_param,
            y_param,
            mode,
            progress,
            task,
        )


if __name__ == "__main__":
    main()
