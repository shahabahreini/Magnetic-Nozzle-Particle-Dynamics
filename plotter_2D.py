import os
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress
import matplotlib.pyplot as plt
import re
from datetime import datetime
from collections import defaultdict
from lib import (
    get_axis_label,
    list_csv_files,
    list_folders,
    extract_parameters_by_file_name,
    find_common_and_varying_params,
)

console = Console()


def plot_csv_files(files, folder, x_param, y_param, mode, progress, task):
    console.print(
        f"\n[green]Generating plot: {get_axis_label(y_param)} vs {get_axis_label(x_param)}[/green]"
    )

    # Check if multiple files are being processed
    is_multi_files = len(files) > 1

    # Find common and varying parameters
    common_params, varying_params, sorted_files = find_common_and_varying_params(files)

    # Update progress bar (10% for reading files)
    progress.update(task, advance=20)

    for file in sorted_files:
        df = pd.read_csv(os.path.join(folder, file))

        x_ = df[x_param]
        y_ = df[y_param]

        # Use only the sorted varying parameters in the legend
        varying_param_str = ", ".join(varying_params[file])

        plt.plot(x_, y_, label=varying_param_str if is_multi_files else None)

    # Update progress bar (40% for generating the plot)
    progress.update(task, advance=40)

    # Title includes common parameters
    common_param_str = ", ".join(
        [f"{get_axis_label(k)}={v}" for k, v in common_params.items()]
    )
    plt.suptitle(f"{get_axis_label(y_param)} vs {get_axis_label(x_param)}", fontsize=12)
    plt.title(common_param_str, loc="right", fontsize=8, color="grey", style="italic")

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
        folders = list_folders()
        folder_choice = IntPrompt.ask(
            "Enter a folder number",
            choices=[str(i) for i in range(1, len(folders) + 1)],
        )
        selected_folder = folders[folder_choice - 1]

        files = list_csv_files(os.path.join(".", selected_folder))

        console.print(
            "\n[bold]Select files to plot (enter numbers separated by space or press Enter to select all):[/bold]"
        )
        file_choice = Prompt.ask("Enter file numbers or press Enter")
        if file_choice:
            try:
                chosen_indices = [int(i) - 1 for i in file_choice.split()]
                selected_files = [
                    files[i] for i in chosen_indices if 0 <= i < len(files)
                ]
                if not selected_files:
                    raise ValueError("Invalid file numbers")
            except ValueError:
                console.print(
                    "[red]Invalid input! Please enter valid file numbers separated by space.[/red]"
                )
                return
        else:
            selected_files = files  # Plot all files if Enter is pressed without input
        folder_path = os.path.join(".", selected_folder)
        mode = "multimode"

    else:
        # Single file selection from the current directory
        folder_path = os.getcwd()  # Current directory
        files = list_csv_files(folder_path)
        file_choice = IntPrompt.ask(
            "Choose a file (enter a number from the list)",
            choices=[str(i) for i in range(1, len(files) + 1)],
        )
        selected_files = [files[file_choice - 1]]
        mode = "singlemode"

    # Parameter options
    x_param_options = ["timestamp", "drho", "dz", "rho", "z", "omega_rho", "omega_z"]
    y_param_options = x_param_options.copy()

    console.print("\n[bold]Select the x-axis parameter:[/bold]")
    for i, param in enumerate(x_param_options, 1):
        console.print(f"{i}. {param}")
    x_param_choice = IntPrompt.ask(
        "Enter your choice",
        choices=[str(i) for i in range(1, len(x_param_options) + 1)],
    )

    console.print("\n[bold]Select the y-axis parameter:[/bold]")
    for i, param in enumerate(y_param_options, 1):
        console.print(f"{i}. {param}")
    y_param_choice = IntPrompt.ask(
        "Enter your choice",
        choices=[str(i) for i in range(1, len(y_param_options) + 1)],
    )

    # Progress bar for plotting (representing file reading, plot generation, saving, and displaying)
    with Progress() as progress:
        task = progress.add_task("[cyan]Plotting...", total=100)
        plot_csv_files(
            selected_files,
            folder_path,
            x_param_options[x_param_choice - 1],
            y_param_options[y_param_choice - 1],
            mode,
            progress,
            task,
        )


if __name__ == "__main__":
    main()
