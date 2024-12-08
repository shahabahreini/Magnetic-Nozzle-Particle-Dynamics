import os
import glob
import pandas as pd
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
import re

console = Console()


def print_styled(text, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{text}{Style.RESET_ALL}")


def search_for_export_csv():
    files = os.listdir()
    csv_files = [file for file in files if file.endswith(".csv")]

    if not csv_files:
        print_styled("No CSV files found in the current directory.", Fore.RED)
        return None

    file_table = [[i + 1, file] for i, file in enumerate(csv_files)]
    print_styled("\nCSV files in current directory:", Fore.CYAN)
    print(tabulate(file_table, headers=["#", "Filename"], tablefmt="fancy_grid"))

    while True:
        choice = input("Choose a file (enter a number from the list): ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(csv_files):
                selected_file = csv_files[choice - 1]
                print_styled(f"Selected file: {selected_file}", Fore.GREEN)
                return selected_file
        except ValueError:
            pass
        print_styled("Invalid choice, please try again.", Fore.RED)


def extract_parameters_by_file_name(fname):
    numbers = {}
    pattern = r"(eps|epsphi|kappa|deltas|beta|alpha|theta|time)(\d+\.\d+(?:e[+-]?\d+)?)"

    for match in re.finditer(pattern, fname):
        key = match.group(1)
        value = float(match.group(2))
        numbers[key] = value

    return numbers


def read_exported_csv_simulation(path_, fname_):
    data = pd.read_csv(path_ + fname_)
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "value1",
            "value2",
            "value3",
            "value4",
            "value5",
            "value6",
        ],
    )
    df.rename(
        columns={
            "value1": "dR",
            "value2": "dphi",
            "value3": "dZ",
            "value4": "R",
            "value5": "phi",
            "value6": "Z",
        },
        inplace=True,
    )
    return df


def read_exported_csv_simulatio_3D(path_, fname_):
    data = pd.read_csv(path_ + fname_)
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "drho",
            "dphi",
            "dz",
            "rho",
            "phi",
            "z",
        ],
    )
    df.rename(
        columns={
            "drho": "dR",
            "dphi": "dphi",
            "dz": "dZ",
            "rho": "R",
            "phi": "phi",
            "z": "Z",
        },
        inplace=True,
    )
    return df


def read_exported_csv_2Dsimulation(path_, fname_):
    fpath = os.path.join(path_, fname_)
    data = pd.read_csv(fpath)
    df = pd.DataFrame(
        data,
        columns=["timestamp", "omega_rho", "omega_z", "rho", "z", "drho", "dz", "dphi"],
    )
    return df


def list_folders(root="."):
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
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if not files:
        console.print(f"[red]No CSV files found in the folder '{folder}'![/red]")
        exit(1)

    table = Table(title=f"\nCSV Files in '{folder}'")
    table.add_column("#", justify="center", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")

    for i, file in enumerate(files, 1):
        table.add_row(str(i), file)

    console.print(table)
    return files


def list_csv_files_noFolder():
    csv_files = glob.glob("*.csv")
    print("\nAvailable CSV files:")
    for idx, file in enumerate(csv_files, 1):
        print(f"{idx}. {file}")
    return csv_files


def find_common_and_varying_params(files):
    """
    Analyze multiple CSV files to find common and varying parameters.

    Args:
        files (list): List of CSV filenames

    Returns:
        tuple: (common_params, varying_params, sorted_files)
            - common_params: dict of parameters that are constant across all files
            - varying_params: dict of parameters that vary between files
            - sorted_files: list of files sorted by varying parameters
    """
    # Extract parameters from all files
    all_params = {}
    for file in files:
        params = extract_parameters_by_file_name(file)
        all_params[file] = params

    # Find common and varying parameters
    if len(files) > 1:
        # Get all parameter keys
        all_keys = set()
        for params in all_params.values():
            all_keys.update(params.keys())

        # Find common and varying parameters
        common_params = {}
        varying_params = {file: [] for file in files}

        for key in all_keys:
            values = [
                all_params[file].get(key) for file in files if key in all_params[file]
            ]

            # If all files have the same value for this parameter
            if len(set(values)) == 1 and len(values) == len(files):
                common_params[key] = values[0]
            else:
                # Add to varying parameters
                for file in files:
                    if key in all_params[file]:
                        varying_params[file].append(f"{key}={all_params[file][key]}")

        # Sort files based on varying parameters
        sorted_files = sorted(
            files,
            key=lambda x: [float(param.split("=")[1]) for param in varying_params[x]],
        )

    else:
        # For single file, all parameters are common
        common_params = all_params[files[0]]
        varying_params = {files[0]: []}
        sorted_files = files

    return common_params, varying_params, sorted_files
