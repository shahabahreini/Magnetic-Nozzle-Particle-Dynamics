#!/usr/bin/env python3

import blessed
import sys
import os
from typing import List, Callable, Dict, Any
import time
import pkg_resources
import subprocess
import importlib.util
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress
from rich import box
from rich.prompt import IntPrompt, Confirm  # Add these to existing imports

# Import functions from your existing files
from plotter_2D import plot_csv_files
from plotter_3D import Plotter
from peakfinder import *
import glob
import psutil


def list_folders():
    """List all folders in the current directory"""
    return [d for d in os.listdir(".") if os.path.isdir(d)]


def list_csv_files(folder_path):
    """List all CSV files in the given folder"""
    return [f for f in os.listdir(folder_path) if f.endswith(".csv")]


term = blessed.Terminal()
console = Console()


def read_requirements():
    """Read requirements from requirements.txt file"""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        console.print("[red]requirements.txt not found![/red]")
        return {}

    requirements = {}
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle version specifiers if present
                package = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                requirements[package.lower()] = package
    return requirements


def check_and_install_requirements():
    """Check and install required packages from requirements.txt"""
    required_packages = read_requirements()

    if not required_packages:
        console.print(
            "[red]No requirements found. Please check requirements.txt file.[/red]"
        )
        input("\nPress Enter to continue...")
        return

    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    missing_packages = []
    installed_packages_info = []

    console.print(
        "\n[bold]Checking package requirements from requirements.txt...[/bold]"
    )

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Checking packages...", total=len(required_packages)
        )

        for package_key, package_name in required_packages.items():
            if package_key not in installed_packages:
                missing_packages.append(package_name)
            else:
                installed_packages_info.append(
                    f"{package_name} (version {installed_packages[package_key]})"
                )
            progress.update(task, advance=1)

    # Display currently installed packages
    if installed_packages_info:
        table = Table(
            title="Currently Installed Packages",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Package", style="cyan")
        table.add_column("Status", style="green")

        for pkg_info in installed_packages_info:
            table.add_row(pkg_info, "✓ Installed")

        console.print(table)

    # Install missing packages
    if missing_packages:
        table = Table(
            title="Missing Packages",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold yellow",
        )
        table.add_column("Package", style="yellow")
        table.add_column("Status", style="yellow")

        for package in missing_packages:
            table.add_row(package, "○ Not installed")

        console.print(table)

        if (
            Prompt.ask(
                "\nWould you like to install missing packages?",
                choices=["y", "n"],
                default="y",
            )
            == "y"
        ):
            console.print("\n[yellow]Installing missing packages...[/yellow]")

            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Installing packages...", total=len(missing_packages)
                )

                for package in missing_packages:
                    try:
                        console.print(f"\nInstalling {package}...")
                        subprocess.check_call(
                            [sys.executable, "-m", "pip", "install", package],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        console.print(
                            f"[green]Successfully installed {package}[/green]"
                        )
                    except subprocess.CalledProcessError as e:
                        console.print(f"[red]Failed to install {package}[/red]")
                        console.print(f"[red]Error: {str(e)}[/red]")
                    progress.update(task, advance=1)

            console.print("\n[green]Package installation completed![/green]")
    else:
        console.print("\n[green]All required packages are already installed![/green]")

    # Final status check
    missing_after_install = []
    for package_key, package_name in required_packages.items():
        if importlib.util.find_spec(package_key.split("[")[0]) is None:
            missing_after_install.append(package_name)

    if missing_after_install:
        console.print(
            "\n[yellow]Warning: Some packages may need manual installation:[/yellow]"
        )
        for package in missing_after_install:
            console.print(f"[yellow]- {package}[/yellow]")
        console.print(
            "\n[yellow]Try installing them manually or check system requirements.[/yellow]"
        )

    console.print("\n[bold green]Dependency check completed![/bold green]")
    input("\nPress Enter to continue...")


class MenuOption:
    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        required_params: Dict[str, Any] = None,
    ):
        self.name = name
        self.function = function
        self.description = description
        self.required_params = required_params or {}


class TerminalUI:
    def __init__(self):
        self.term = term
        self.current_selection = 0
        self.options = [
            MenuOption(
                "Package Management",
                check_and_install_requirements,
                "Check and install required packages",
                {},
            ),
            MenuOption(
                "2D Plotting",
                self.run_2d_plotter,
                "Generate various 2D plots from CSV files",
                {},
            ),
            MenuOption(
                "3D Trajectory Plotting",
                self.run_3d_plotter,
                "Generate 3D trajectory visualizations",
                {},
            ),
            MenuOption(
                "Peak Analysis",
                self.run_peak_analysis,
                "Analyze peaks and adiabatic conditions",
                {},
            ),
            MenuOption(
                "Amplitude Analysis",
                self.run_amplitude_analysis,
                "Analyze amplitude variations over time",
                {},
            ),
        ]

    def run_2d_plotter(self):
        """Interactive 2D plotting interface"""
        # Display folder/file selection
        console.print("\n[bold]2D Plot Generation[/bold]")
        console.print("1. Select files from a folder")
        console.print("2. Select a single file")
        choice = Prompt.ask("Choose an option", choices=["1", "2"])

        # Get folder path
        if choice == "1":
            folders = list_folders()
            table = Table(title="Available Folders")
            table.add_column("Number", style="cyan")
            table.add_column("Folder", style="magenta")
            for i, folder in enumerate(folders, 1):
                table.add_row(str(i), folder)
            console.print(table)

            folder_idx = IntPrompt.ask("Select folder number", default=1)
            folder_path = folders[folder_idx - 1]
        else:
            folder_path = "."

        # List available files
        files = list_csv_files(folder_path)
        table = Table(title="Available CSV Files")
        table.add_column("Number", style="cyan")
        table.add_column("Filename", style="magenta")
        for i, file in enumerate(files, 1):
            table.add_row(str(i), file)
        console.print(table)

        # File selection
        if choice == "1":
            file_nums = Prompt.ask(
                "Enter file numbers (comma-separated) or press Enter for all"
            )
            if file_nums.strip():
                selected_files = [files[int(i) - 1] for i in file_nums.split(",")]
            else:
                selected_files = files
        else:
            file_idx = IntPrompt.ask("Select file number", default=1)
            selected_files = [files[file_idx - 1]]

        # Parameter selection
        params = ["timestamp", "drho", "dz", "rho", "z", "omega_rho", "omega_z"]
        table = Table(title="Available Parameters")
        table.add_column("Number", style="cyan")
        table.add_column("Parameter", style="magenta")
        for i, param in enumerate(params, 1):
            table.add_row(str(i), param)
        console.print(table)

        x_param_idx = IntPrompt.ask("Select x-axis parameter", default=1)
        y_param_idx = IntPrompt.ask("Select y-axis parameter", default=2)

        # Execute plotting
        plot_csv_files(
            selected_files,
            folder_path,
            params[x_param_idx - 1],
            params[y_param_idx - 1],
            "interactive",
            Progress(),
            Progress().add_task("Plotting...", total=100),
        )

    def run_3d_plotter(self):
        """Interactive 3D plotting interface"""
        console.print("\n[bold]3D Trajectory Plotting[/bold]")

        # Get folder path
        folders = list_folders()
        table = Table(title="Available Folders")
        table.add_column("Number", style="cyan")
        table.add_column("Folder", style="magenta")
        for i, folder in enumerate(folders, 1):
            table.add_row(str(i), folder)
        console.print(table)

        folder_idx = IntPrompt.ask("Select folder number", default=1)
        folder_path = folders[folder_idx - 1]

        # List and select files
        files = list_csv_files(folder_path)
        table = Table(title="Available CSV Files")
        table.add_column("Number", style="cyan")
        table.add_column("Filename", style="magenta")
        for i, file in enumerate(files, 1):
            table.add_row(str(i), file)
        console.print(table)

        file_nums = Prompt.ask(
            "Enter file numbers (comma-separated) or press Enter for all"
        )
        if file_nums.strip():
            selected_files = [files[int(i) - 1] for i in file_nums.split(",")]
        else:
            selected_files = files

        # Plot options
        coord_system = Prompt.ask(
            "Select coordinate system",
            choices=["cartesian", "cylindrical"],
            default="cartesian",
        )
        use_scatter = Confirm.ask("Use scatter plot?", default=True)
        use_time_color = Confirm.ask("Use time-based coloring?", default=True)
        show_projections = Confirm.ask("Show projections?", default=False)

        # Execute plotting
        Plotter(
            selected_files,
            folder_path,
            plot_type="3d",
            coord_system=coord_system,
            use_scatter=use_scatter,
            use_time_color=use_time_color,
            show_projections=show_projections,
        )

    def run_peak_analysis(self):
        """Interactive peak analysis interface with enhanced UX"""

        def display_header():
            console.print(
                Panel(
                    "[bold blue]Peak Analysis Interface[/bold blue]\n"
                    "[dim]Analyze particle trajectory peaks and adiabatic conditions[/dim]",
                    box=box.DOUBLE,
                    style="bold",
                    border_style="blue",
                )
            )

        def load_configuration():
            """Load and validate configuration settings"""
            try:
                config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
                config = Configuration(config_path)
                return config
            except Exception as e:
                console.print(f"[red]Error loading configuration: {str(e)}[/red]")
                return None

        def select_input_files(config):
            """Enhanced file selection interface"""
            if not config.is_multi_files:
                with Progress() as progress:
                    task = progress.add_task(
                        "[cyan]Scanning for CSV files...", total=100
                    )
                    chosen_csv = search_for_export_csv()
                    progress.update(task, completed=100)

                    if not chosen_csv:
                        console.print("[red]No CSV file selected.[/red]")
                        return None

                    return os.path.basename(chosen_csv).replace(".csv", "")
            else:
                return "multi_plot"

        def display_analysis_options(config):
            """Display available analysis options in an enhanced table"""
            table = Table(
                title="Available Analysis Options",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
                title_style="bold blue",
            )

            table.add_column("Option", style="cyan", justify="center")
            table.add_column("Analysis Type", style="magenta")
            table.add_column("Status", style="green", justify="center")
            table.add_column("Description", style="yellow")

            options = [
                (
                    "1",
                    "Integral Analysis",
                    config.calculate_integral,
                    "Calculate and analyze integral properties of the trajectory",
                ),
                (
                    "2",
                    "Traditional Magnetic Moment",
                    config.calculate_traditional_magneticMoment,
                    "Analyze magnetic moment using traditional methods",
                ),
                (
                    "3",
                    "Amplitude Analysis",
                    config.show_amplitude_analysis,
                    "Analyze amplitude variations over time",
                ),
                (
                    "4",
                    "Run All Enabled Analyses",
                    True,
                    "Execute all enabled analysis types",
                ),
            ]

            for opt, name, enabled, desc in options:
                status = (
                    "[green]✓ Enabled[/green]" if enabled else "[red]✗ Disabled[/red]"
                )
                table.add_row(opt, name, status, desc)

            console.print(table)

        def run_analysis(choice, config, chosen_csv):
            """Execute selected analysis with enhanced progress tracking"""
            analyses = {
                1: ("Integral Analysis", plotter, config.calculate_integral),
                2: (
                    "Magnetic Moment Analysis",
                    perform_adiabatic_calculations,
                    config.calculate_traditional_magneticMoment,
                ),
                3: (
                    "Amplitude Analysis",
                    plot_amplitude_analysis_separate,
                    config.show_amplitude_analysis,
                ),
            }

            try:
                with Progress() as progress:
                    if choice == 4:
                        # Run all enabled analyses
                        total_enabled = sum(
                            1 for _, _, enabled in analyses.values() if enabled
                        )
                        if total_enabled == 0:
                            console.print(
                                "[yellow]No analyses are currently enabled in configuration.[/yellow]"
                            )
                            return False

                        overall_progress = progress.add_task(
                            "[blue]Overall Progress", total=total_enabled
                        )

                        for analysis_name, func, enabled in analyses.values():
                            if enabled:
                                task = progress.add_task(
                                    f"[cyan]Running {analysis_name}...", total=100
                                )

                                # Run the analysis
                                if analysis_name == "Integral Analysis":
                                    func(config.target_folder, chosen_csv)
                                else:
                                    func(chosen_csv)

                                progress.update(task, completed=100)
                                progress.update(overall_progress, advance=1)

                                console.print(
                                    f"[green]✓ {analysis_name} completed[/green]"
                                )

                    else:
                        # Run single analysis
                        analysis_name, func, enabled = analyses.get(
                            choice, (None, None, False)
                        )

                        if not enabled:
                            console.print(
                                f"[yellow]The selected analysis is not enabled in configuration.[/yellow]"
                            )
                            return False

                        task = progress.add_task(
                            f"[cyan]Running {analysis_name}...", total=100
                        )

                        # Run the analysis
                        if analysis_name == "Integral Analysis":
                            func(config.target_folder, chosen_csv)
                        else:
                            func(chosen_csv)

                        progress.update(task, completed=100)
                        console.print(f"[green]✓ {analysis_name} completed[/green]")

                return True

            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Error during analysis:[/red]\n{str(e)}",
                        title="Error",
                        border_style="red",
                    )
                )
                return False

        # Main execution flow
        display_header()

        # Load configuration
        config = load_configuration()
        if not config:
            return

        # Select input files with external progress bar
        with Progress() as progress:
            chosen_csv = None
            if not config.is_multi_files:
                chosen_csv = search_for_export_csv(
                    external_progress=progress  # Pass the progress bar
                )
                if not chosen_csv:
                    return
                chosen_csv = os.path.basename(chosen_csv).replace(".csv", "")
            else:
                chosen_csv = "multi_plot"

        # Select input files
        chosen_csv = select_input_files(config)
        if not chosen_csv:
            return

        # Display analysis options
        display_analysis_options(config)

        # Get user choice
        choice = IntPrompt.ask(
            "\nSelect analysis type", choices=["1", "2", "3", "4"], default="4"
        )

        # Run analysis
        if run_analysis(choice, config, chosen_csv):
            console.print(
                Panel(
                    "[green]Analysis completed successfully![/green]\n"
                    "[dim]Press Enter to return to main menu[/dim]",
                    box=box.ROUNDED,
                )
            )
        else:
            console.print(
                Panel(
                    "[yellow]Analysis completed with warnings or errors.[/yellow]\n"
                    "[dim]Press Enter to return to main menu[/dim]",
                    box=box.ROUNDED,
                )
            )

    def run_amplitude_analysis(self):
        """Interactive amplitude analysis interface"""
        console.print("\n[bold]Amplitude Analysis[/bold]")

        # Get file path
        folders = list_folders()
        table = Table(title="Available Folders")
        table.add_column("Number", style="cyan")
        table.add_column("Folder", style="magenta")
        for i, folder in enumerate(folders, 1):
            table.add_row(str(i), folder)
        console.print(table)

        folder_idx = IntPrompt.ask("Select folder number", default=1)
        folder_path = folders[folder_idx - 1]

        files = list_csv_files(folder_path)
        table = Table(title="Available CSV Files")
        table.add_column("Number", style="cyan")
        table.add_column("Filename", style="magenta")
        for i, file in enumerate(files, 1):
            table.add_row(str(i), file)
        console.print(table)

        file_idx = IntPrompt.ask("Select file number", default=1)
        selected_file = files[file_idx - 1]

        # Execute analysis
        plot_amplitude_analysis_separate(folder_path, selected_file)

    def run(self):
        """Run the terminal UI"""
        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            while True:
                # Clear screen and reset cursor
                print(self.term.home + self.term.clear)

                # Display header and menu
                self.display_header()
                self.display_menu()

                # Get user input
                key = self.term.inkey()

                if key.name == "KEY_UP":
                    self.current_selection = (self.current_selection - 1) % len(
                        self.options
                    )
                elif key.name == "KEY_DOWN":
                    self.current_selection = (self.current_selection + 1) % len(
                        self.options
                    )
                elif key.name == "KEY_ENTER":
                    # Clear screen for the selected option
                    print(self.term.clear)

                    try:
                        # Execute the selected option's function
                        selected_option = self.options[self.current_selection]
                        selected_option.function()

                        # Wait for user acknowledgment before returning to menu
                        console.print(
                            "\n[cyan]Press Enter to return to main menu...[/cyan]"
                        )
                        input()
                    except Exception as e:
                        console.print(f"\n[red]Error: {str(e)}[/red]")
                        console.print(
                            "\n[cyan]Press Enter to return to main menu...[/cyan]"
                        )
                        input()
                elif key == "q":
                    break

        # Clear screen when exiting
        print(self.term.clear)

    def display_header(self):
        """Display enhanced application header"""
        header_text = Panel(
            "[bold blue]Particle Trajectory Analysis Tool[/bold blue]\n"
            "[dim]Navigation: [cyan]↑/↓[/cyan] arrows to move, "
            "[cyan]Enter[/cyan] to select, [cyan]q[/cyan] to quit, "
            "[cyan]f[/cyan] to filter files[/dim]",
            box=box.DOUBLE,
            style="bold",
            border_style="blue",
        )
        console.print(header_text)

    def display_menu(self):
        """Display enhanced main menu"""
        menu_table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title="Main Menu",
            title_style="bold blue",
        )

        menu_table.add_column("", style="cyan", justify="center", width=3)
        menu_table.add_column("Function", style="green", width=25)
        menu_table.add_column("Description", style="yellow")

        for i, option in enumerate(self.options):
            marker = "►" if i == self.current_selection else " "
            menu_table.add_row(marker, option.name, option.description)

        console.print(menu_table)

    def display_status_bar(self):
        """Display status bar with system information"""
        status = Panel(
            f"[cyan]System Status:[/cyan] Running | "
            f"[green]Memory Usage:[/green] {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB | "
            f"[yellow]Files Loaded:[/yellow] {len(glob.glob('*.csv'))} CSV",
            box=box.HORIZONTALS,
            style="dim",
        )
        console.print(status)

    def filter_files(self, files: List[str], pattern: str = "") -> List[str]:
        """
        Filter files based on search pattern

        Args:
            files (List[str]): List of files to filter
            pattern (str): Search pattern

        Returns:
            List[str]: Filtered list of files
        """
        if not pattern:
            return files

        filtered = [f for f in files if pattern.lower() in f.lower()]

        if not filtered:
            console.print(f"[yellow]No files matching pattern '{pattern}'[/yellow]")

        return filtered

    def prompt_file_selection(self, files: List[str]) -> str:
        """
        Enhanced file selection prompt with filtering

        Args:
            files (List[str]): List of files to choose from

        Returns:
            str: Selected filename
        """
        while True:
            console.print("\n[bold cyan]File Selection[/bold cyan]")
            console.print("Enter 'f' to filter, 'q' to quit, or file number to select")

            choice = Prompt.ask("Choice")

            if choice.lower() == "f":
                pattern = Prompt.ask("Enter search pattern")
                filtered = self.filter_files(files, pattern)
                if filtered:
                    return self.prompt_file_selection(filtered)
            elif choice.lower() == "q":
                return None
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(files):
                        return files[idx]
                except ValueError:
                    pass

            console.print("[red]Invalid selection[/red]")

    def run_with_progress(self, func, *args, **kwargs):
        """
        Run a function with progress indication
        """
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=100)
            try:
                result = func(*args, **kwargs)
                progress.update(task, completed=100)
                return result
            except Exception as e:
                progress.update(task, completed=100, description="[red]Error![/red]")
                console.print(f"[red]Error: {str(e)}[/red]")
                return None


def main():
    ui = TerminalUI()
    ui.run()


if __name__ == "__main__":
    main()
