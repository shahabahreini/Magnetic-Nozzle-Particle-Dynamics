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
    Enhanced folder and file listing function with pagination and detailed information.
    Lists folders first, then CSV files in the current directory.
    """
    import os
    from datetime import datetime
    import math
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    def get_item_info(path):
        """Get detailed information about a folder or file"""
        stats = os.stat(path)
        is_dir = os.path.isdir(path)
        
        info = {
            "name": os.path.basename(path),
            "modified": datetime.fromtimestamp(stats.st_mtime),
            "path": str(path),
            "is_dir": is_dir
        }
        
        if is_dir:
            try:
                items = list(os.scandir(path))
                info["num_files"] = len([x for x in items if x.is_file()])
                info["num_folders"] = len([x for x in items if x.is_dir()])
                
                total_size = 0
                for item in Path(path).rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
                info["size"] = total_size
            except PermissionError:
                info["num_files"] = -1
                info["num_folders"] = -1
                info["size"] = -1
        else:
            info["size"] = stats.st_size
            
        return info

    def format_size(size):
        """Format size in human-readable format"""
        if size < 0:
            return "N/A"
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def display_items_table(items, current_page, total_pages):
        """Display folders and files in a rich formatted table"""
        console.clear()

        console.print(
            Panel(
                Text(f"Contents of '{os.path.abspath(root)}'", style="bold white", justify="center"),
                style="blue",
            )
        )

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title=f"Page {current_page + 1} of {total_pages}",
        )

        table.add_column("#", style="cyan", justify="center", width=4)
        table.add_column("Type", style="yellow", width=6)
        table.add_column("Name", style="green")
        table.add_column("Size", style="yellow", justify="right")
        table.add_column("Files", style="blue", justify="right", width=8)
        table.add_column("Folders", style="magenta", justify="right", width=8)
        table.add_column("Last Modified", style="cyan")

        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(items))

        for idx, item in enumerate(items[start_idx:end_idx], start=start_idx + 1):
            if item["is_dir"]:
                files_str = "N/A" if item["num_files"] < 0 else str(item["num_files"])
                folders_str = "N/A" if item["num_folders"] < 0 else str(item["num_folders"])
                item_type = "[blue]DIR[/blue]"
            else:
                files_str = "-"
                folders_str = "-"
                item_type = "[green]CSV[/green]"
            
            table.add_row(
                str(idx),
                item_type,
                item["name"],
                format_size(item["size"]),
                files_str,
                folders_str,
                item["modified"].strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    try:
        # Get both folders and CSV files
        items = []
        
        # First get folders
        for item in os.scandir(root):
            if item.is_dir():
                items.append(get_item_info(item.path))
                
        # Then get CSV files
        for item in os.scandir(root):
            if item.is_file() and item.name.lower().endswith('.csv'):
                items.append(get_item_info(item.path))

        if not items:
            console.print(
                Panel(
                    "[red]No folders or CSV files found in the current directory![/red]",
                    title="Error",
                    border_style="red",
                )
            )
            return None, None

        # Sort items: folders first (alphabetically), then files (alphabetically)
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        current_page = 0
        total_pages = math.ceil(len(items) / per_page)

        while True:
            display_items_table(items, current_page, total_pages)

            nav_options = []
            if current_page > 0:
                nav_options.append("[cyan]'p'[/cyan] Previous")
            if current_page < total_pages - 1:
                nav_options.append("[cyan]'n'[/cyan] Next")
            nav_options.extend(["[cyan]'q'[/cyan] Quit", "or enter number to select"])

            console.print(
                Panel(" | ".join(nav_options), title="Navigation", border_style="green")
            )

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
                if 0 <= idx < len(items):
                    selected_item = items[idx]
                    
                    if selected_item["is_dir"]:
                        console.print(
                            Panel(
                                f"[green]Selected Folder: {selected_item['name']}[/green]\n"
                                f"[blue]Path: {selected_item['path']}[/blue]\n"
                                f"[magenta]Files: {selected_item['num_files']} | Folders: {selected_item['num_folders']}[/magenta]",
                                title="Folder Selected",
                                border_style="green",
                            )
                        )
                        # Return folder path and None for file path
                        return selected_item["path"], None
                    else:
                        console.print(
                            Panel(
                                f"[green]Selected File: {selected_item['name']}[/green]\n"
                                f"[blue]Size: {format_size(selected_item['size'])}[/blue]\n"
                                f"[magenta]Modified: {selected_item['modified'].strftime('%Y-%m-%d %H:%M')}[/magenta]",
                                title="File Selected",
                                border_style="green",
                            )
                        )
                        # Return None for folder path and file path
                        return None, selected_item["path"]
                else:
                    console.print(
                        Panel(
                            "[red]Invalid number. Please try again.[/red]",
                            border_style="red",
                        )
                    )
                    console.input("Press Enter to continue...")
            except ValueError:
                if choice not in ["n", "p", "q"]:
                    console.print(
                        Panel(
                            "[red]Invalid input. Please try again.[/red]",
                            border_style="red",
                        )
                    )
                    console.input("Press Enter to continue...")

    except Exception as e:
        console.print(
            Panel(
                f"[red]Error: {str(e)}[/red]",
                title="Error",
                border_style="red",
            )
        )
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


def list_comparison_files(folder_path, comparison_type, file_role):
    """
    Generic CSV file listing function for solution comparisons.

    Args:
        folder_path (str): Folder path to list CSV files from
        comparison_type (str): Type of comparison ("1D_2D" or "2D_3D")
        file_role (str): Role of the file being selected ("reference", "approximation", "2D", "3D")

    Returns:
        tuple: (selected_file, all_files) or (None, None) if cancelled
    """
    console = Console()

    # Define selection contexts for different comparison types
    selection_info = {
        # For 1D to 2D/3D comparisons
        "1D_2D": {
            "reference": {
                "title": "Select Reference (2D/3D) Solution File",
                "description": (
                    "[yellow]Please select the reference solution file (2D/3D).[/yellow]\n"
                    "[dim]This should be the high-fidelity solution to compare against.[/dim]"
                ),
                "color": "blue",
                "expected_type": "2D/3D",
            },
            "approximation": {
                "title": "Select 1D Approximation File",
                "description": (
                    "[yellow]Please select the 1D approximation solution file.[/yellow]\n"
                    "[dim]This will be compared against the reference solution.[/dim]"
                ),
                "color": "green",
                "expected_type": "1D",
            },
        },
        # For 2D to 3D comparisons
        "2D_3D": {
            "2D": {
                "title": "Select 2D Solution File",
                "description": (
                    "[yellow]Please select the 2D solution file for comparison.[/yellow]\n"
                    "[dim]This will be compared against the 3D solution.[/dim]"
                ),
                "color": "blue",
                "expected_type": "2D",
            },
            "3D": {
                "title": "Select 3D Solution File",
                "description": (
                    "[yellow]Please select the 3D solution file for comparison.[/yellow]\n"
                    "[dim]This will be compared against the 2D solution.[/dim]"
                ),
                "color": "green",
                "expected_type": "3D",
            },
        },
    }

    # Get context for current selection
    context = selection_info[comparison_type][file_role]

    # Get all CSV files and their information
    files = []
    for f in os.listdir(folder_path):
        if f.endswith(".csv"):
            full_path = os.path.join(folder_path, f)
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
    per_page = 10
    total_pages = (len(files) + per_page - 1) // per_page

    while True:
        console.clear()

        # Print header with context
        console.print(
            Panel(
                context["description"],
                title=context["title"],
                border_style=context["color"],
            )
        )

        # Create and populate table
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style=context["color"],
            title=f"Page {current_page + 1} of {total_pages}",
        )

        table.add_column("#", style="cyan", justify="center", width=4)
        table.add_column("Filename", style="green")
        table.add_column("Size", style="yellow", justify="right")
        table.add_column("Last Modified", style="magenta")
        table.add_column("Type", style="cyan", justify="center")

        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(files))

        for idx, file in enumerate(files[start_idx:end_idx], start=start_idx + 1):
            # Determine file type from filename
            file_type = "Unknown"
            if "1d" in file["name"].lower():
                file_type = "1D"
            elif "2d" in file["name"].lower():
                file_type = "2D"
            elif "3d" in file["name"].lower():
                file_type = "3D"

            table.add_row(
                str(idx),
                file["name"],
                format_size(file["size"]),
                file["modified"].strftime("%Y-%m-%d %H:%M"),
                file_type,
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
            Panel(" | ".join(nav_options), title="Navigation", border_style="cyan")
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

                # Determine file type and show warning if needed
                file_type = "Unknown"
                if "1d" in selected_file["name"].lower():
                    file_type = "1D"
                elif "2d" in selected_file["name"].lower():
                    file_type = "2D"
                elif "3d" in selected_file["name"].lower():
                    file_type = "3D"

                warning_message = ""
                if file_type != "Unknown" and file_type != context["expected_type"]:
                    warning_message = f"\n[red]Warning: This appears to be a {file_type} file, but you're selecting a {context['expected_type']} solution.[/red]"

                console.print(
                    Panel(
                        f"[green]Selected: {selected_file['name']}[/green]\n"
                        f"[blue]Size: {format_size(selected_file['size'])}[/blue]\n"
                        f"[magenta]Modified: {selected_file['modified'].strftime('%Y-%m-%d %H:%M')}[/magenta]\n"
                        f"[cyan]Detected Type: {file_type}[/cyan]"
                        f"{warning_message}\n\n"
                        "[yellow]Press Enter to confirm or 'r' to reselect[/yellow]",
                        title="Confirm Selection",
                        border_style="green",
                    )
                )

                confirm = console.input().lower().strip()
                if confirm != "r":
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


def list_items(root=".", select_type="file", file_extension=".csv", file_keywords=None, per_page=40):
    """
    Enhanced folder and file listing function with pagination and detailed information.
    
    Args:
        root (str): Root directory to list items from
        select_type (str): Type of selection - "file" or "folder"
        file_extension (str): File extension to filter (e.g., ".csv")
        file_keywords (list): List of keywords to filter files (e.g., ["1D", "2D", "3D"])
        per_page (int): Number of items to display per page
    
    Returns:
        str: Selected path (file or folder path depending on select_type)
    """
    import os
    from datetime import datetime
    import math
    from pathlib import Path
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    def matches_keywords(filename):
        """Check if filename matches any of the keywords"""
        if not file_keywords:
            return True
        return any(keyword.lower() in filename.lower() for keyword in file_keywords)

    def get_item_info(path):
        """Get detailed information about a folder or file"""
        stats = os.stat(path)
        is_dir = os.path.isdir(path)
        
        info = {
            "name": os.path.basename(path),
            "modified": datetime.fromtimestamp(stats.st_mtime),
            "path": str(path),
            "is_dir": is_dir
        }
        
        if is_dir:
            try:
                items = list(os.scandir(path))
                info["num_files"] = len([x for x in items if x.is_file()])
                info["num_folders"] = len([x for x in items if x.is_dir()])
                
                # Count only files that match both extension and keywords
                target_files = [x for x in items if x.is_file() and 
                              x.name.lower().endswith(file_extension) and 
                              matches_keywords(x.name)]
                info["num_target_files"] = len(target_files)
                
                total_size = 0
                for item in Path(path).rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
                info["size"] = total_size
            except PermissionError:
                info["num_files"] = -1
                info["num_folders"] = -1
                info["num_target_files"] = -1
                info["size"] = -1
        else:
            info["size"] = stats.st_size
            
        return info

    def format_size(size):
        """Format size in human-readable format"""
        if size < 0:
            return "N/A"
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    def display_items_table(items, current_page, total_pages, show_files=True):
        """Display folders and optionally files in a rich formatted table"""
        console.clear()

        # Create header text
        header_text = f"Contents of '{os.path.abspath(root)}'"
        if file_keywords:
            header_text += f"\nFiltering for: {', '.join(file_keywords)}"

        console.print(
            Panel(
                Text(header_text, style="bold white", justify="center"),
                style="blue",
            )
        )

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title=f"Page {current_page + 1} of {total_pages}",
        )

        table.add_column("#", style="cyan", justify="center", width=4)
        table.add_column("Type", style="yellow", width=6)
        table.add_column("Name", style="green")
        table.add_column("Size", style="yellow", justify="right")
        if show_files:
            filtered_label = f"Filtered {file_extension.upper()}"
            table.add_column(filtered_label, style="blue", justify="right", width=10)
        table.add_column("Files", style="blue", justify="right", width=8)
        table.add_column("Folders", style="magenta", justify="right", width=8)
        table.add_column("Last Modified", style="cyan")

        start_idx = current_page * per_page
        end_idx = min(start_idx + per_page, len(items))

        for idx, item in enumerate(items[start_idx:end_idx], start=start_idx + 1):
            if item["is_dir"]:
                files_str = "N/A" if item["num_files"] < 0 else str(item["num_files"])
                folders_str = "N/A" if item["num_folders"] < 0 else str(item["num_folders"])
                target_files_str = "N/A" if item["num_target_files"] < 0 else str(item["num_target_files"])
                item_type = "[blue]DIR[/blue]"
                
                row_data = [
                    str(idx),
                    item_type,
                    item["name"],
                    format_size(item["size"])
                ]
                if show_files:
                    row_data.append(target_files_str)
                row_data.extend([
                    files_str,
                    folders_str,
                    item["modified"].strftime("%Y-%m-%d %H:%M")
                ])
            else:
                item_type = f"[green]{file_extension.upper()[1:]}[/green]"
                row_data = [
                    str(idx),
                    item_type,
                    item["name"],
                    format_size(item["size"])
                ]
                if show_files:
                    row_data.append("-")
                row_data.extend([
                    "-",
                    "-",
                    item["modified"].strftime("%Y-%m-%d %H:%M")
                ])
            
            table.add_row(*row_data)

        console.print(table)

    def browse_location(current_path):
        """Browse the current location and handle navigation"""
        while True:
            # Get items in current location
            items = []
            
            # Get folders
            for item in os.scandir(current_path):
                if item.is_dir():
                    items.append(get_item_info(item.path))
            
            # Get files if we're selecting files
            if select_type == "file":
                for item in os.scandir(current_path):
                    if (item.is_file() and 
                        item.name.lower().endswith(file_extension) and 
                        matches_keywords(item.name)):
                        items.append(get_item_info(item.path))

            if not items:
                message = "[red]No "
                if select_type == "folder":
                    message += "folders"
                else:
                    message += f"{file_extension} files"
                    if file_keywords:
                        message += f" matching keywords: {', '.join(file_keywords)}"
                message += " found in this directory![/red]"
                
                console.print(
                    Panel(
                        message,
                        title="Error",
                        border_style="red",
                    )
                )
                if current_path != root:
                    return "back"
                return None

            # Sort items: folders first (alphabetically), then files (alphabetically)
            items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
            
            current_page = 0
            total_pages = math.ceil(len(items) / per_page)

            while True:
                display_items_table(items, current_page, total_pages, show_files=(select_type == "file"))

                nav_options = []
                if current_page > 0:
                    nav_options.append("[cyan]'p'[/cyan] Previous")
                if current_page < total_pages - 1:
                    nav_options.append("[cyan]'n'[/cyan] Next")
                if current_path != root:
                    nav_options.append("[cyan]'b'[/cyan] Back")
                nav_options.extend(["[cyan]'q'[/cyan] Quit", "or enter number to select"])

                console.print(
                    Panel(" | ".join(nav_options), title="Navigation", border_style="green")
                )

                choice = console.input("\nYour choice: ").lower().strip()

                if choice == "q":
                    return None
                elif choice == "b" and current_path != root:
                    return "back"
                elif choice == "n" and current_page < total_pages - 1:
                    current_page += 1
                    continue
                elif choice == "p" and current_page > 0:
                    current_page -= 1
                    continue

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(items):
                        selected_item = items[idx]
                        
                        if selected_item["is_dir"]:
                            if select_type == "folder":
                                console.print(
                                    Panel(
                                        f"[green]Selected Folder: {selected_item['name']}[/green]\n"
                                        f"[blue]Path: {selected_item['path']}[/blue]\n"
                                        f"[magenta]Files: {selected_item['num_files']} | Folders: {selected_item['num_folders']}[/magenta]",
                                        title="Folder Selected",
                                        border_style="green",
                                    )
                                )
                                return selected_item["path"]
                            else:
                                # Navigate into the folder
                                result = browse_location(selected_item["path"])
                                if result == "back":
                                    break
                                if result is not None:
                                    return result
                        else:
                            if select_type == "file":
                                console.print(
                                    Panel(
                                        f"[green]Selected File: {selected_item['name']}[/green]\n"
                                        f"[blue]Size: {format_size(selected_item['size'])}[/blue]\n"
                                        f"[magenta]Modified: {selected_item['modified'].strftime('%Y-%m-%d %H:%M')}[/magenta]",
                                        title="File Selected",
                                        border_style="green",
                                    )
                                )
                                return selected_item["path"]
                    else:
                        console.print(
                            Panel(
                                "[red]Invalid number. Please try again.[/red]",
                                border_style="red",
                            )
                        )
                        console.input("Press Enter to continue...")
                except ValueError:
                    if choice not in ["n", "p", "b", "q"]:
                        console.print(
                            Panel(
                                "[red]Invalid input. Please try again.[/red]",
                                border_style="red",
                            )
                        )
                        console.input("Press Enter to continue...")

    try:
        return browse_location(root)
    except Exception as e:
        console.print(
            Panel(
                f"[red]Error: {str(e)}[/red]",
                title="Error",
                border_style="red",
            )
        )
        return None
