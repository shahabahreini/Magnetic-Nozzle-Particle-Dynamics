import os
import sys
import importlib
from blessed import Terminal
from typing import Dict, Any, Callable
import yaml
import pandas as pd
import numpy as np
import pkg_resources
import subprocess
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress
from rich import box
from pathlib import Path

import matplotlib.pyplot as plt
from modules import (
    search_for_export_csv,
    extract_parameters_by_file_name,
    list_csv_files,
    list_folders,
    find_common_and_varying_params,  # This is now properly imported
)
from peakfinder import (
    plotter,
    plot_amplitude_analysis_separate,
    plotter_adiabatic_invariance_check,
    plot_eta_fluctuations,
    read_exported_csv_2Dsimulation,
    Configuration,
)
from plotter_2D import plot_csv_files, select_plotting_parameters
from plotter_3D import Plotter, display_available_parameters, get_parameter_from_input


term = Terminal()
config = Configuration(os.path.join(os.path.dirname(__file__), "config.yaml"))


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


def print_header():
    print(term.clear + term.bold_green("=" * 50))
    print(term.bold_yellow(f"Welcome to the Analysis Scripts Menu"))
    print(term.bold_white(f"Developed by Shahab Bahreini Jangjoo"))
    print(term.bold_green("=" * 50))


def print_footer():
    print(term.bold_green("=" * 50))
    print(term.bold_white("Thank you for using the system. Goodbye!"))
    print(term.bold_green("=" * 50))


def get_csv_file():
    """Helper function to get CSV file from user"""
    print(term.clear)
    print(term.bold_blue("\nSelect CSV file:"))

    # List all CSV files in current directory
    csv_files = [f for f in os.listdir() if f.endswith(".csv")]

    if not csv_files:
        print(term.red("\nNo CSV files found in current directory!"))
        return None

    for i, file in enumerate(csv_files, 1):
        print(term.cyan(f"[{i}] {file}"))

    try:
        choice = int(input(term.bold("\nEnter file number: "))) - 1
        if 0 <= choice < len(csv_files):
            return csv_files[choice].replace(".csv", "")
        else:
            print(term.red("\nInvalid selection!"))
            return None
    except ValueError:
        print(term.red("\nInvalid input!"))
        return None


def run_adiabatic_condition_analysis():
    """Run the adiabatic condition analysis"""
    fname = get_csv_file()
    if fname:
        try:
            plotter(os.getcwd(), fname)
            plt.show()
        except Exception as e:
            print(term.red(f"\nError during analysis: {str(e)}"))


def run_amplitude_analysis():
    """Run the amplitude analysis"""
    fname = get_csv_file()
    if fname:
        try:
            plot_amplitude_analysis_separate(os.getcwd(), fname, True)
            plt.show()
        except Exception as e:
            print(term.red(f"\nError during analysis: {str(e)}"))


def run_adiabatic_invariance_check():
    """Run the adiabatic invariance check"""
    fname = get_csv_file()
    if fname:
        try:
            plotter_adiabatic_invariance_check(os.getcwd(), fname, True)
            plt.show()
        except Exception as e:
            print(term.red(f"\nError during analysis: {str(e)}"))


def run_eta_fluctuations_analysis():
    """Run the η fluctuations analysis"""
    fname = get_csv_file()
    if fname:
        try:
            # First read the data
            df = read_exported_csv_2Dsimulation(os.getcwd(), fname + ".csv")
            plot_eta_fluctuations(df, fname, True)
            plt.show()
        except Exception as e:
            print(term.red(f"\nError during analysis: {str(e)}"))


def configure_settings():
    """Configure analysis settings"""
    print(term.clear)
    print(term.bold_blue("\nCurrent Settings:"))

    settings = {
        "is_multi_files": ("Enable multi-file analysis", bool),
        "based_on_guiding_center": ("Use guiding center for calculations", bool),
        "show_extremums_peaks": ("Show extremum peaks in plots", bool),
        "extremum_of": ("Variable to find extremums", str),
        "share_x_axis": ("Share x-axis in plots", bool),
    }

    for key, (desc, _) in settings.items():
        current = getattr(config, key, "Not set")
        print(term.cyan(f"{desc}: {current}"))

    if input(term.bold("\nWould you like to modify settings? (y/n): ")).lower() == "y":
        try:
            for key, (desc, type_) in settings.items():
                value = input(
                    term.bold(f"\nEnter new value for {desc} ({type_.__name__}): ")
                )
                if type_ == bool:
                    setattr(config, key, value.lower() in ("true", "yes", "1", "y"))
                else:
                    setattr(config, key, type_(value))

            config.save()
            print(term.bold_green("\nSettings saved successfully!"))
        except Exception as e:
            print(term.red(f"\nError saving settings: {str(e)}"))


def peakfinder_menu():
    """Submenu for peakfinder functionality"""
    options = {
        "1": ("Run Adiabatic Condition Analysis", run_adiabatic_condition_analysis),
        "2": ("Run Amplitude Analysis", run_amplitude_analysis),
        "3": ("Run Adiabatic Invariance Check", run_adiabatic_invariance_check),
        "4": ("Run η Fluctuations Analysis", run_eta_fluctuations_analysis),
        "5": ("Configure Settings", configure_settings),
        "b": ("Back to Main Menu", None),
    }

    while True:
        print(term.clear)
        print(term.bold_blue("\nPeak Finder Analysis Menu:"))
        for key, (desc, _) in options.items():
            print(term.cyan(f"[{key}] {desc}"))

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "b":
            return
        elif choice in options and options[choice][1]:
            try:
                options[choice][1]()
                input(term.bold_green("\nPress Enter to continue..."))
            except Exception as e:
                print(term.red(f"\nError: {str(e)}"))
                input(term.bold_red("\nPress Enter to continue..."))


def plotter_2d_menu():
    """Submenu for 2D plotter functionality"""
    while True:
        print(term.clear)
        print(term.bold_blue("\n2D Plotter Menu:"))

        # Mode selection
        console.print("[bold]Select mode of operation:[/bold]")
        console.print("1. Select files from a folder")
        console.print("2. Select a single file from the current directory")
        console.print("b. Back to Main Menu")

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "b":
            return
        elif choice in ["1", "2"]:
            try:
                if choice == "1":
                    # Folder selection mode
                    console.print("[bold]Select a folder containing CSV files:[/bold]")
                    selected_folder, selected_files = list_folders()
                    folder_path = os.path.join(".", selected_folder)
                    mode = "multimode"
                else:
                    # Single file selection from current directory
                    folder_path = os.getcwd()
                    selected_file, all_files = list_csv_files(folder_path)
                    selected_files = [selected_file]
                    mode = "singlemode"

                # Parameter selection
                param_options = [
                    "timestamp",
                    "drho",
                    "dz",
                    "rho",
                    "z",
                    "omega_rho",
                    "omega_z",
                ]
                x_param, y_param = select_plotting_parameters(param_options)

                # Create progress bar
                with Progress() as progress:
                    task = progress.add_task("[cyan]Generating plot...", total=100)
                    plot_csv_files(
                        selected_files,
                        folder_path,
                        x_param,
                        y_param,
                        mode,
                        progress,
                        task,
                    )

                input(term.bold_green("\nPress Enter to continue..."))
            except Exception as e:
                print(term.red(f"\nError: {str(e)}"))
                input(term.bold_red("\nPress Enter to continue..."))
        else:
            print(term.red("Invalid choice. Please try again."))
            input(term.bold_red("\nPress Enter to continue..."))


def plotter_3d_menu():
    """Submenu for 3D plotter functionality"""
    while True:
        print(term.clear)
        print(term.bold_blue("\n3D Plotter Menu:"))
        console.print(Panel("3D Trajectory Plotter", style="bold magenta"))

        options = {
            "1": "Select files and plot",
            "2": "Configure plot settings",
            "b": "Back to Main Menu",
        }

        for key, desc in options.items():
            print(term.cyan(f"[{key}] {desc}"))

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "b":
            return
        elif choice == "1":
            try:
                # List CSV files in the folder
                folder_path, selected_files = list_folders()

                if not selected_files:
                    console.print("[red]No CSV files found![/red]")
                    continue

                # Read the first CSV file to get available parameters
                df = pd.read_csv(os.path.join(folder_path, selected_files[0]))

                # Ask user for plot type
                plot_type = Prompt.ask(
                    "Enter the plot type", choices=["2d", "3d"], default="3d"
                )

                if plot_type == "2d":
                    display_available_parameters(df)
                    x_param = get_parameter_from_input(df, "Enter parameter for x-axis")
                    y_param = get_parameter_from_input(df, "Enter parameter for y-axis")
                    coord_system = "cartesian"
                else:
                    coord_system = Prompt.ask(
                        "Enter coordinate system",
                        choices=["cartesian", "cylindrical"],
                        default="cartesian",
                    )
                    x_param = y_param = None

                # Additional plot settings
                use_scatter = Confirm.ask("Use scatter plot?", default=True)
                use_time_color = Confirm.ask("Use time-based coloring?", default=True)
                show_projections = Confirm.ask("Show projections?", default=False)

                # Generate the plot
                Plotter(
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

                input(term.bold_green("\nPress Enter to continue..."))
            except Exception as e:
                print(term.red(f"\nError: {str(e)}"))
                input(term.bold_red("\nPress Enter to continue..."))
        else:
            print(term.red("Invalid choice. Please try again."))
            input(term.bold_red("\nPress Enter to continue..."))


def main_menu():
    options = {
        "1": ("Package Management", check_and_install_requirements),
        "2": ("Peak Finder Analysis", peakfinder_menu),
        "3": ("2D Plotter", plotter_2d_menu),
        "4": ("3D Plotter", plotter_3d_menu),
        "q": ("Quit", None),
    }

    while True:
        print_header()
        for key, (desc, _) in options.items():
            print(term.cyan(f"[{key}] {desc}"))

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "q":
            print_footer()
            sys.exit()
        elif choice in options and options[choice][1]:
            options[choice][1]()
        else:
            print(term.red("Invalid choice. Please try again."))
            input(term.bold_red("\nPress Enter to continue..."))


if __name__ == "__main__":
    main_menu()
