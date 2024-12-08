import os
import glob
import pandas as pd
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
import re

from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich import box
from rich.text import Text
from rich.progress import Progress
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

console = Console()


def print_styled(text, color=Fore.WHITE, style=Style.NORMAL):
    print(f"{style}{color}{text}{Style.RESET_ALL}")


def search_for_export_csv(
    directory: str = ".",
    pattern: str = None,
    sort_by: str = "name",
    external_progress: Progress = None,
) -> Optional[str]:
    """
    Enhanced CSV file search and selection interface.

    Args:
        directory (str): Directory to search for CSV files
        pattern (str): Optional filter pattern
        sort_by (str): Sorting criterion ('name', 'date', 'size')
        external_progress (Progress): Optional external progress bar

    Returns:
        Optional[str]: Selected CSV filename or None if cancelled
    """

    def get_file_info(filepath: Path) -> Dict:
        """Get detailed file information"""
        stats = filepath.stat()
        return {
            "name": filepath.name,
            "size": stats.st_size,
            "modified": stats.st_mtime,
            "path": filepath,
        }

    def display_files(
        files: List[Dict], current_page: int = 0, per_page: int = 10
    ) -> None:
        """Display files in a paginated table"""
        start_idx = current_page * per_page
        page_files = files[start_idx : start_idx + per_page]
        total_pages = (len(files) + per_page - 1) // per_page

        table = Table(
            title=f"CSV Files ({current_page + 1}/{total_pages})",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("#", style="cyan", justify="right")
        table.add_column("Filename", style="green")
        table.add_column("Size", style="blue", justify="right")
        table.add_column("Modified", style="yellow")

        for idx, file_info in enumerate(page_files, start=start_idx + 1):
            size_str = f"{file_info['size'] / 1024:.1f} KB"
            date_str = datetime.fromtimestamp(file_info["modified"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            table.add_row(str(idx), file_info["name"], size_str, date_str)

        console.print(table)

        if total_pages > 1:
            console.print(
                "\n[dim]Navigation: [cyan]n[/cyan] next page, "
                "[cyan]p[/cyan] previous page, "
                "[cyan]f[/cyan] filter, "
                "[cyan]s[/cyan] sort, "
                "[cyan]q[/cyan] quit[/dim]"
            )

    def filter_files(files: List[Dict], pattern: str) -> List[Dict]:
        """Filter files based on pattern"""
        if not pattern:
            return files
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            return [f for f in files if regex.search(f["name"])]
        except re.error:
            pattern = pattern.lower()
            return [f for f in files if pattern in f["name"].lower()]

    def sort_files(files: List[Dict], criterion: str) -> List[Dict]:
        """Sort files based on criterion"""
        if criterion == "date":
            return sorted(files, key=lambda x: x["modified"], reverse=True)
        elif criterion == "size":
            return sorted(files, key=lambda x: x["size"], reverse=True)
        else:  # sort by name
            return sorted(files, key=lambda x: x["name"].lower())

    try:
        # Scan files with progress handling
        path = Path(directory)

        if external_progress:
            # Use external progress bar if provided
            task = external_progress.add_task(
                "[cyan]Scanning for CSV files...", total=100
            )
            all_files = [get_file_info(f) for f in path.glob("*.csv") if f.is_file()]
            external_progress.update(task, completed=100)
        else:
            # Create temporary progress bar if none provided
            with Progress() as progress:
                task = progress.add_task("[cyan]Scanning for CSV files...", total=100)
                all_files = [
                    get_file_info(f) for f in path.glob("*.csv") if f.is_file()
                ]
                progress.update(task, completed=100)

        if not all_files:
            console.print(
                Panel(
                    "[yellow]No CSV files found in the specified directory.[/yellow]",
                    title="Notice",
                    border_style="yellow",
                )
            )
            return None

        # Apply initial filtering if pattern provided
        files = filter_files(all_files, pattern)

        # Apply initial sorting
        files = sort_files(files, sort_by)

        current_page = 0
        per_page = 10

        while True:
            console.clear()
            console.print(
                Panel(
                    "[bold blue]CSV File Selection[/bold blue]\n"
                    "[dim]Enter file number to select, or use navigation commands[/dim]",
                    box=box.DOUBLE,
                    style="bold",
                    border_style="blue",
                )
            )

            display_files(files, current_page, per_page)

            choice = Prompt.ask(
                "\nEnter choice", default="", show_default=False
            ).lower()

            if choice == "q":
                return None

            elif choice == "n":
                if (current_page + 1) * per_page < len(files):
                    current_page += 1

            elif choice == "p":
                if current_page > 0:
                    current_page -= 1

            elif choice == "f":
                pattern = Prompt.ask("Enter filter pattern")
                files = filter_files(all_files, pattern)
                current_page = 0

            elif choice == "s":
                sort_options = {"1": "name", "2": "date", "3": "size"}
                console.print("\n[cyan]Sort by:[/cyan]")
                for key, value in sort_options.items():
                    console.print(f"{key}. {value.capitalize()}")

                sort_choice = Prompt.ask(
                    "Select sorting criterion",
                    choices=list(sort_options.keys()),
                    default="1",
                )
                files = sort_files(files, sort_options[sort_choice])
                current_page = 0

            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(files):
                        selected_file = files[idx]
                        console.print(
                            Panel(
                                f"[green]Selected: {selected_file['name']}[/green]",
                                border_style="green",
                            )
                        )
                        return str(selected_file["path"])
                    else:
                        console.print("[red]Invalid file number[/red]")
                except ValueError:
                    console.print("[red]Invalid input[/red]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return None


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


def list_folders(root=".", per_page=40):
    """
    Enhanced folder listing function with pagination and detailed information.
    Lists folders first, then CSV files in selected folder.

    Args:
        root (str): Root directory to list folders from
        per_page (int): Number of items to display per page

    Returns:
        tuple: (selected_folder_path, selected_file_path) or (None, None) if cancelled
    """
    import os
    from datetime import datetime
    import math
    from pathlib import Path

    def clear_screen():
        """Clear the terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")

    def get_folder_info(folder_path):
        """Get detailed information about a folder"""
        path = Path(folder_path)
        stats = path.stat()

        try:
            items = list(path.iterdir())
            num_files = len([x for x in items if x.is_file()])
            num_folders = len([x for x in items if x.is_dir()])
        except PermissionError:
            num_files = num_folders = -1

        total_size = 0
        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
        except PermissionError:
            total_size = -1

        return {
            "name": path.name,
            "modified": stats.st_mtime,
            "num_files": num_files,
            "num_folders": num_folders,
            "size": total_size,
            "path": str(path),
        }

    def format_size(size):
        """Format size in human-readable format"""
        if size < 0:
            return "N/A"
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def print_header(title):
        clear_screen()
        print("\n" + "=" * 100)
        print(" " * ((100 - len(title)) // 2) + title)
        print("=" * 100 + "\n")

    def print_folder_table(folders, current_page=0, per_page=10):
        """Print folders in a formatted table"""
        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(folders))
        total_pages = math.ceil(len(folders) / per_page)

        print(f"\nPage {current_page + 1} of {total_pages}")
        print("-" * 100)
        print(
            f"{'#':4} {'Folder Name':<30} {'Size':>12} {'Files':>8} {'Folders':>8} {'Last Modified':>20}"
        )
        print("-" * 100)

        for idx, folder in enumerate(folders[start_idx:end_idx], start=start_idx + 1):
            modified_time = datetime.fromtimestamp(folder["modified"]).strftime(
                "%Y-%m-%d %H:%M"
            )
            files_str = "N/A" if folder["num_files"] < 0 else str(folder["num_files"])
            folders_str = (
                "N/A" if folder["num_folders"] < 0 else str(folder["num_folders"])
            )
            size_str = format_size(folder["size"])

            print(
                f"{idx:3}. {folder['name']:<30} {size_str:>12} {files_str:>8} {folders_str:>8} {modified_time:>20}"
            )

        print("-" * 100)

    def list_csv_files(folder_path, current_page=0, per_page=10):
        """List CSV files in the selected folder"""
        csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
        if not csv_files:
            print("\nNo CSV files found in this folder!")
            return None

        csv_files.sort()
        total_pages = math.ceil(len(csv_files) / per_page)
        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(csv_files))

        print_header(f"CSV Files in '{os.path.basename(folder_path)}'")
        print(f"Page {current_page + 1} of {total_pages}")
        print("-" * 100)
        print(f"{'#':4} {'Filename':<90}")
        print("-" * 100)

        for idx, file in enumerate(csv_files[start_idx:end_idx], start=start_idx + 1):
            print(f"{idx:3}. {file:<90}")

        print("-" * 100)

        if total_pages > 1:
            print("\nNavigation:")
            print("  'n' - Next page")
            print("  'p' - Previous page")
            print("  'b' - Back to folder selection")
            print("  'q' - Quit")
            print("  Or enter file number to select")
        else:
            print("\nEnter file number to select, 'b' for back, or 'q' to quit")

        while True:
            choice = input("\nYour choice: ").lower().strip()

            if choice == "q":
                return None
            elif choice == "b":
                return "back"
            elif choice == "n" and current_page < total_pages - 1:
                return list_csv_files(folder_path, current_page + 1, per_page)
            elif choice == "p" and current_page > 0:
                return list_csv_files(folder_path, current_page - 1, per_page)

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    return [csv_files[idx]]
                else:
                    print("\nInvalid file number. Please try again.")
            except ValueError:
                if choice not in ["n", "p", "b", "q"]:
                    print("\nInvalid input. Please try again.")

    try:
        while True:
            # Get folders and their information
            folders = []
            for item in os.scandir(root):
                if item.is_dir():
                    folders.append(get_folder_info(item.path))

            if not folders:
                print("\nNo folders found in the current directory!")
                return None, None

            # Sort folders by name
            folders.sort(key=lambda x: x["name"].lower())

            current_page = 0
            total_pages = math.ceil(len(folders) / per_page)

            print_header("Folder Selection")
            print_folder_table(folders, current_page, per_page)

            if total_pages > 1:
                print("\nNavigation:")
                print("  'n' - Next page")
                print("  'p' - Previous page")
                print("  'q' - Quit")
                print("  Or enter folder number to select")
            else:
                print("\nEnter folder number to select, or 'q' to quit")

            choice = input("\nYour choice: ").lower().strip()

            if choice == "q":
                return None, None
            elif choice == "n" and current_page < total_pages - 1:
                current_page += 1
                continue
            elif choice == "p" and current_page > 0:
                current_page -= 1
                continue

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(folders):
                    selected_folder = folders[idx]
                    folder_path = selected_folder["path"]
                    # List CSV files in the selected folder
                    selected_file = list_csv_files(folder_path)
                    if selected_file == "back":
                        continue
                    return folder_path, selected_file
                else:
                    print("\nInvalid folder number. Please try again.")
            except ValueError:
                if choice not in ["n", "p", "q"]:
                    print("\nInvalid input. Please try again.")

    except Exception as e:
        print(f"\nError: {str(e)}")
        return None, None


def get_file_info(filepath):
    """Get detailed information about a file."""
    stats = os.stat(filepath)
    return {
        "size": stats.st_size,
        "modified": datetime.fromtimestamp(stats.st_mtime),
        "created": datetime.fromtimestamp(stats.st_ctime),
    }


def format_size(size):
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def list_csv_files(folder, per_page=10):
    """
    Enhanced CSV file listing function with pagination and detailed information.

    Args:
        folder (str): Folder path to list CSV files from
        per_page (int): Number of files to display per page

    Returns:
        tuple: (selected_file, all_files) or (None, None) if cancelled
    """
    console = Console()

    # Get all CSV files and their information
    files = []
    for f in os.listdir(folder):
        if f.endswith(".csv"):
            full_path = os.path.join(folder, f)
            file_info = get_file_info(full_path)
            files.append({"name": f, "path": full_path, **file_info})

    if not files:
        console.print(
            Panel(
                "[red]No CSV files found in this folder![/red]",
                title="Error",
                border_style="red",
            )
        )
        return None, None

    # Sort files by name
    files.sort(key=lambda x: x["name"].lower())
    current_page = 0
    total_pages = (len(files) + per_page - 1) // per_page

    while True:
        console.clear()

        # Print header
        console.print(
            Panel(
                Text(
                    f"CSV Files in '{os.path.basename(folder)}'",
                    style="bold white",
                    justify="center",
                ),
                style="blue",
            )
        )

        # Create and populate table
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title=f"Page {current_page + 1} of {total_pages}",
        )

        table.add_column("#", style="cyan", justify="center", width=4)
        table.add_column("Filename", style="green")
        table.add_column("Size", style="yellow", justify="right")
        table.add_column("Last Modified", style="magenta")

        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(files))

        for idx, file in enumerate(files[start_idx:end_idx], start=start_idx + 1):
            table.add_row(
                str(idx),
                file["name"],
                format_size(file["size"]),
                file["modified"].strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

        # Navigation help
        nav_options = []
        if current_page > 0:
            nav_options.append("[cyan]'p'[/cyan] Previous")
        if current_page < total_pages - 1:
            nav_options.append("[cyan]'n'[/cyan] Next")
        nav_options.extend(["[cyan]'q'[/cyan] Quit", "or enter file number"])

        console.print(
            Panel(" | ".join(nav_options), title="Navigation", border_style="green")
        )

        # Get user input
        choice = console.input("\nYour choice: ").lower().strip()

        if choice == "q":
            return None, None
        elif choice == "n" and current_page < total_pages - 1:
            current_page += 1
            continue
        elif choice == "p" and current_page > 0:
            current_page -= 1
            continue

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                selected_file = files[idx]
                console.print(
                    Panel(
                        f"[green]Selected: {selected_file['name']}[/green]\n"
                        + f"[blue]Size: {format_size(selected_file['size'])}[/blue]\n"
                        + f"[magenta]Modified: {selected_file['modified'].strftime('%Y-%m-%d %H:%M')}[/magenta]",
                        title="File Selected",
                        border_style="green",
                    )
                )
                return selected_file["path"], [f["path"] for f in files]
            else:
                console.print(
                    Panel(
                        "[red]Invalid file number. Please try again.[/red]",
                        border_style="red",
                    )
                )
                input("Press Enter to continue...")
        except ValueError:
            if choice not in ["n", "p", "q"]:
                console.print(
                    Panel(
                        "[red]Invalid input. Please try again.[/red]",
                        border_style="red",
                    )
                )
                input("Press Enter to continue...")


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
