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

console = Console()


# Helper function to extract parameters from the filename
def extract_parameters_by_file_name(fname):
    numbers = {}

    # Regex to match scientific notation and float values
    pattern = r"(eps|epsphi|kappa|deltas|beta|alpha|theta|time)(\d+\.\d+(?:e[+-]?\d+)?)"

    for match in re.finditer(pattern, fname):
        key = match.group(1)
        # Convert the matched string to a float
        value = float(match.group(2))
        numbers[key] = value

    return numbers


# Helper function to find common and varying parameters, sort the varying ones
def find_common_and_varying_params(files):
    all_params = [(file, extract_parameters_by_file_name(file)) for file in files]
    common_params = {}
    varying_params = defaultdict(list)

    # Extract parameter names
    param_names = set(all_params[0][1].keys())

    # Find common parameters
    for param in param_names:
        param_values = [params[param] for _, params in all_params]
        if all(v == param_values[0] for v in param_values):
            # If the parameter is the same for all files, it's common
            common_params[param] = param_values[0]
        else:
            # Otherwise, it's a varying parameter
            for file, params in all_params:
                varying_params[file].append(f"{get_axis_label(param)}={params[param]}")

    # Sort files based on one of the varying parameters (e.g., 'eps')
    sorted_files = sorted(all_params, key=lambda x: x[1].get('eps', 0))

    # Sort varying parameters based on file order
    sorted_varying_params = {file: varying_params[file] for file, _ in sorted_files}

    return common_params, sorted_varying_params, [file for file, _ in sorted_files]


def list_folders(root='.'):
    # List all directories in the root folder
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    if not folders:
        console.print("[red]No folders found in the current directory![/red]")
        exit(1)

    table = Table(title="Available Folders")
    table.add_column("#", justify="center", style="cyan", no_wrap=True)
    table.add_column("Folder", style="magenta")

    for i, folder in enumerate(folders, 1):
        table.add_row(str(i), folder)

    console.print(table)
    return folders


def list_csv_files(folder):
    # List all CSV files in the selected folder
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        console.print(f"[red]No CSV files found in the folder '{folder}'![/red]")
        exit(1)

    table = Table(title=f"CSV Files in '{folder}'")
    table.add_column("#", justify="center", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")

    for i, file in enumerate(files, 1):
        table.add_row(str(i), file)

    console.print(table)
    return files


def get_axis_label(param):
    labels = {
        'rho': r'$\tilde{R}$',
        'z': r'$\tilde{Z}$',
        'drho': r'$d\tilde{R}/d\tau$',
        'dz': r'$d\tilde{Z}/d\tau$',
        'timestamp': r'$\tau$',
        'omega_rho': r'$\omega_{\tilde{R}}$',
        'omega_z': r'$\omega_{\tilde{Z}}$',
        'eps': r'$\epsilon$',
        'epsphi': r'$\epsilon_{\phi}$',
        'kappa': r'$\kappa$',
        'deltas': r'$\delta_s$',
        'beta': r'$\beta$',
        'alpha': r'$\alpha$',
        'theta': r'$\theta$',
        'time': r'$\tau$'
    }
    return labels.get(param, param)


def plot_csv_files(files, folder, x_param, y_param, mode, progress, task):
    console.print(f"\n[green]Generating plot: {get_axis_label(y_param)} vs {get_axis_label(x_param)}[/green]")

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
    common_param_str = ", ".join([f"{get_axis_label(k)}={v}" for k, v in common_params.items()])
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
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    plot_filename = f"{mode}-{current_time}.png"

    # Create plots folder if it doesn't exist
    plots_folder = os.path.join('.', 'plots')
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
        folder_choice = IntPrompt.ask("Enter a folder number", choices=[str(i) for i in range(1, len(folders) + 1)])
        selected_folder = folders[folder_choice - 1]

        files = list_csv_files(os.path.join('.', selected_folder))

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
        folder_path = os.path.join('.', selected_folder)
        mode = "multimode"

    else:
        # Single file selection from the current directory
        folder_path = os.getcwd()  # Current directory
        files = list_csv_files(folder_path)
        file_choice = IntPrompt.ask("Choose a file (enter a number from the list)",
                                    choices=[str(i) for i in range(1, len(files) + 1)])
        selected_files = [files[file_choice - 1]]
        mode = "singlemode"

    # Parameter options
    x_param_options = ["timestamp", "drho", "dz", "rho", "z", "omega_rho", "omega_z"]
    y_param_options = x_param_options.copy()

    console.print("\n[bold]Select the x-axis parameter:[/bold]")
    for i, param in enumerate(x_param_options, 1):
        console.print(f"{i}. {param}")
    x_param_choice = IntPrompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(x_param_options) + 1)])

    console.print("\n[bold]Select the y-axis parameter:[/bold]")
    for i, param in enumerate(y_param_options, 1):
        console.print(f"{i}. {param}")
    y_param_choice = IntPrompt.ask("Enter your choice", choices=[str(i) for i in range(1, len(y_param_options) + 1)])

    # Progress bar for plotting (representing file reading, plot generation, saving, and displaying)
    with Progress() as progress:
        task = progress.add_task("[cyan]Plotting...", total=100)
        plot_csv_files(selected_files, folder_path, x_param_options[x_param_choice - 1],
                       y_param_options[y_param_choice - 1], mode, progress, task)


if __name__ == "__main__":
    main()
