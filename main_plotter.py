import os
import sys
import importlib
from blessed import Terminal
from typing import Dict, Any, Callable
import pandas as pd
import numpy as np
import pkg_resources
import subprocess
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm
from rich import box
from rich.box import ROUNDED, SQUARE
from pathlib import Path

import matplotlib.pyplot as plt
from modules import (
    search_for_export_csv,
    extract_parameters_by_file_name,
    list_csv_files,
    list_folders,
    find_common_and_varying_params,
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
from plotter_3D import Plotter_2d3d, display_available_parameters, get_parameter_from_input
from plotter_comparison_1D_exact import (
    list_csv_files_noFolder,
    select_file_by_number,
    find_first_difference,
    create_fancy_annotation,
    save_plots_with_timestamp,
)
from plotter_comparison_2D_3D import plot_comparison
from modules.file_utils import list_comparison_files, list_items
from plotter_energy_momentum_conservasion import CONFIG, plotter_conservation


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
    print(term.clear)
    print_header()
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
    print(term.bold_white("Thank you for using the Plotter UI. Goodbye!"))
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
    console.print(
        Panel(
            "[bold blue]Peakfinder functionality:[/bold blue]",
            style="bold white",
            expand=False,
        )
    )
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
        print_header()
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
    options = {
        "1": ("Select files from a folder", "multimode"),
        "2": ("Select a single file from the current directory", "singlemode"),
        "b": ("Back to Main Menu", None),
    }

    while True:
        print(term.clear)
        print_header()
        console.print(
            Panel(
                "[bold blue]2D Plotter Menu:[/bold blue]",
                style="bold white",
                expand=False,
            )
        )

        for key, (desc, _) in options.items():
            print(term.cyan(f"[{key}] {desc}"))

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "b":
            return

        if choice in options and options[choice][1]:
            mode = options[choice][1]
            try:
                if mode == "multimode":
                    console.print("[bold]Select a folder containing CSV files:[/bold]")
                    selected_folder, selected_files = list_folders()
                    folder_path = os.path.join(".", selected_folder)
                else:
                    folder_path = os.getcwd()
                    selected_file = list_items(root=".", select_type="file", file_keywords=["2D"])
                    selected_files = [selected_file]

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
                console.print(f"[red]\nError: {str(e)}[/red]")
                input(term.bold_red("\nPress Enter to continue..."))
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")
            input(term.bold_red("\nPress Enter to continue..."))


def plotter_3d_menu():
    """Submenu for 3D plotter functionality"""
    options = {
        "1": ("Select files from a folder", "multimode"),
        "2": ("Select a single file from the current directory", "singlemode"),
        "b": ("Back to Main Menu", None),
    }

    while True:
        print(term.clear)
        print_header()
        console.print(
            Panel(
                "[bold blue]3D Plotter Menu:[/bold blue]",
                style="bold white",
                expand=False,
            )
        )

        for key, (desc, _) in options.items():
            print(term.cyan(f"[{key}] {desc}"))

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "b":
            return

        if choice in options and options[choice][1]:
            mode = options[choice][1]
            try:
                if mode == "multimode":
                    console.print("[bold]Select a folder containing CSV files:[/bold]")
                    selected_folder, selected_files = list_folders()
                    folder_path = os.path.join(".", selected_folder)
                else:
                    folder_path = os.getcwd()
                    selected_file = list_items(root=".", select_type="file", file_keywords=["3D"])
                    selected_files = [selected_file]

                # Read the first CSV file to get available parameters
                df = pd.read_csv(os.path.join(folder_path, selected_files[0]))

                # Get plot configuration
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

                with Progress() as progress:
                    task = progress.add_task("[cyan]Generating plot...", total=100)
                    
                    try:
                        Plotter_2d3d(
                            selected_files,
                            folder_path,
                            x_param,
                            y_param,
                            plot_type,
                            coord_system,
                            use_scatter,
                            use_time_color,
                            show_projections,
                            progress=progress,
                            task=task
                        )
                        
                        console.print('\n')
                        console.print(
                            Panel(
                                "[green]✓ Plot generated successfully![/green]",
                                title="Success",
                                border_style="green",
                            )
                        )
                        
                    except Exception as e:
                        console.print('\n')
                        console.print(
                            Panel(
                                f"[red]Error during plot generation:",
                                title="Error",
                                border_style="red",
                            )
                        )
                        raise

                input(term.bold_green("\nPress Enter to continue..."))
            except Exception as e:
                console.print(f"[red]\nError: {str(e)}[/red]")
                input(term.bold_red("\nPress Enter to continue..."))
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")
            input(term.bold_red("\nPress Enter to continue..."))


def plotter_comparison_menu():
    """Submenu for Comparison Approximated 1D to Exact 2D/3D Solution"""
    options = {
        "1": ("Run Comparison 1D to Exact 2D/3D", run_comparison_1d_to_exact),
        "2": ("Run Comparison 2D to Exact 3D", run_2d_3d_comparison),
        "b": ("Back to Main Menu", None),
    }

    while True:

        print(term.clear)
        print_header()
        console.print(
            Panel(
                "[bold blue]Comparison Approximated 1D to Exact 2D/3D Menu:[/bold blue]",
                style="bold white",
                expand=False,
            )
        )

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
                console.print(f"[red]\nError: {str(e)}[/red]")
                input(term.bold_red("\nPress Enter to continue..."))
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")
            input(term.bold_red("\nPress Enter to continue..."))


def run_comparison_1d_to_exact():
    """Run the comparison analysis between 1D and Exact 2D/3D solutions"""
    parameter_mapping = {
        "eps": r"$\epsilon$",
        "epsphi": r"$\epsilon_\phi$",
        "kappa": r"$\kappa$",
        "deltas": r"$\delta_s$",
        "beta": r"$\beta_0$",
        "alpha": r"$\alpha_0$",
        "theta": r"$\theta_0$",
        "time": r"$\tau$",
    }

    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Running comparison analysis...", total=100)

            # Get the current folder path
            folder_path = os.getcwd()

            # File selection for 2D reference data
            progress.update(
                task,
                description="[cyan]Selecting 2D reference solution file...",
                advance=10,
            )
            file_2d, all_files = list_comparison_files(
                folder_path, comparison_type="1D_2D", file_role="reference"
            )

            if not file_2d:
                console.print(
                    Panel(
                        "[red]Process cancelled: No file selected for 2D reference data.[/red]\n"
                        "[yellow]Please ensure 2D solution files are present in the directory.[/yellow]",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            # File selection for 1D comparison data
            progress.update(
                task,
                description="[cyan]Selecting 1D approximation file...",
                advance=10,
            )
            file_1d, _ = list_comparison_files(
                folder_path, comparison_type="1D_2D", file_role="approximation"
            )

            if not file_1d:
                console.print(
                    Panel(
                        "[red]Process cancelled: No file selected for 1D approximation data.[/red]\n"
                        "[yellow]Please ensure 1D solution files are present in the directory.[/yellow]",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            # Validate file selections
            if file_2d == file_1d:
                console.print(
                    Panel(
                        "[red]Error: Same file selected for both 2D reference and 1D approximation.[/red]\n"
                        "[yellow]Please select different files for comparison.[/yellow]",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            # Display selected files information
            console.print(
                Panel(
                    f"[bold green]Selected files for comparison:[/bold green]\n\n"
                    f"[blue]2D Reference:[/blue] {os.path.basename(file_2d)}\n"
                    f"[green]1D Approximation:[/green] {os.path.basename(file_1d)}",
                    title="Comparison Setup",
                    border_style="cyan",
                )
            )

            if not file_1d:
                console.print("[red]No file selected for 1D comparison data.[/red]")
                return

            console.print(
                Panel(
                    f"[bold green]Selected files:[/bold green]\n"
                    f"2D (reference): {os.path.basename(file_2d)}\n"
                    f"1D (comparison): {os.path.basename(file_1d)}",
                    style="bold white",
                    expand=False,
                )
            )

            # Data loading and processing
            progress.update(task, advance=20)
            df_2d = pd.read_csv(file_2d)
            df_1d = pd.read_csv(file_1d)

            time_2d = df_2d["timestamp"]
            z_2d = df_2d["z"]
            time_1d = df_1d["timestamp"]
            z_1d = df_1d["z"]

            progress.update(task, advance=20)
            interpolated_z_1d = np.interp(time_2d, time_1d, z_1d)

            # Difference analysis
            progress.update(task, advance=10)
            result = find_first_difference(time_2d, z_2d, interpolated_z_1d, 10)
            if result is None:
                result = find_first_difference(time_2d, z_2d, interpolated_z_1d, 1)

            # Plot creation
            progress.update(task, advance=20)
            fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

            ax.plot(
                time_2d,
                z_2d,
                label="2D Exact Z Solution (Reference)",
                color="#2E86C1",
                linewidth=2,
                alpha=0.8,
            )
            ax.plot(
                time_2d,
                interpolated_z_1d,
                label="1D Approximate Z Solution",
                color="#E67E22",
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )

            if result is not None:
                annotation_text = (
                    f"Error > {result['threshold']}% Difference\n"
                    f"────────────────────\n"
                    f"Time: {result['time']:.3f}\n"
                    r"$Z_{2D}$" + f"(Exact): {result['z_2d']:.3f}\n"
                    r"$Z_{1D}$" + f"(Approximated): {result['z_1d']:.3f}\n"
                    f"Δ: {result['difference']:.2f}%"
                )

                create_fancy_annotation(
                    fig,
                    ax,
                    xy=(result["time"], result["z_2d"]),
                    text=annotation_text,
                    xytext=(
                        result["time"] - (time_2d.max() - time_2d.min()) * 0.1,
                        result["z_2d"] + (z_2d.max() - z_2d.min()) * 0.1,
                    ),
                )

                ax.plot(
                    result["time"],
                    result["z_2d"],
                    "o",
                    color="#E74C3C",
                    markersize=8,
                    alpha=0.8,
                )

                title = "Trajectory Comparison - Reduced 1D Equation vs Exact Solution"
            else:
                title = "Trajectory Comparison (No significant differences found)"

            ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
            ax.set_xlabel(r"$\tau$", labelpad=10)
            ax.set_ylabel(r"$\tilde z$", labelpad=10)
            ax.legend(
                loc="upper right",
                framealpha=0.95,
                edgecolor="#666666",
                fancybox=True,
                shadow=True,
            )

            for spine in ax.spines.values():
                spine.set_color("#CCCCCC")
                spine.set_linewidth(0.8)

            parameters = extract_parameters_by_file_name(file_2d)
            param_text = "\n".join(
                f"{parameter_mapping.get(key, key)}: {value}"
                for key, value in parameters.items()
            )

            ax.text(
                0.02,
                0.95,
                "Simulation Parameters:\n" + param_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    edgecolor="#CCCCCC",
                    alpha=0.9,
                    linewidth=0.5,
                ),
            )

            # Final plot adjustments and saving
            progress.update(task, advance=10)
            plt.tight_layout()
            save_plots_with_timestamp(fig, "1D_2D_z_solutions_comparison")
            plt.show()

    except Exception as e:
        console.print(f"[red]\nError: {str(e)}[/red]")
        raise  # Re-raise the exception to be caught by the menu handler


def calculate_error_statistics(df_1d, df_ref):
    """
    Calculate error statistics between 1D approximation and reference solution.

    Args:
        df_1d (pd.DataFrame): 1D approximation data
        df_ref (pd.DataFrame): Reference solution data

    Returns:
        dict: Dictionary containing error statistics
    """
    # Note: Implement this function based on your specific needs
    # This is a placeholder implementation
    return {
        "max_error": 0.0,  # Maximum relative error
        "avg_error": 0.0,  # Average relative error
        "rms_error": 0.0,  # Root mean square error
    }


def run_2d_3d_comparison():
    """
    Run the comparison analysis between 2D and 3D solutions.
    Handles file selection, data validation, and visualization of the comparison.
    """
    console = Console()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            main_task = progress.add_task(
                "[cyan]Running 2D-3D comparison analysis...", total=100
            )

            # Get the current folder path
            folder_path = os.getcwd()

            # File selection for 2D data
            progress.update(
                main_task, description="[cyan]Selecting 2D solution file...", advance=10
            )
            file_2d, all_files = list_comparison_files(
                folder_path, comparison_type="2D_3D", file_role="2D"
            )

            if not file_2d:
                console.print(
                    Panel(
                        "[red]Process cancelled: No file selected for 2D data.[/red]",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            # File selection for 3D data
            progress.update(
                main_task, description="[cyan]Selecting 3D solution file...", advance=10
            )
            file_3d, _ = list_comparison_files(
                folder_path, comparison_type="2D_3D", file_role="3D"
            )

            if not file_3d:
                console.print(
                    Panel(
                        "[red]Process cancelled: No file selected for 3D data.[/red]",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            # Validate file selections
            if file_2d == file_3d:
                console.print(
                    Panel(
                        "[red]Error: Same file selected for both 2D and 3D data.[/red]\n"
                        "[yellow]Please select different files for comparison.[/yellow]",
                        title="Error",
                        border_style="red",
                    )
                )
                return

            # Display selected files information
            console.print(
                Panel(
                    f"[bold green]Selected files for comparison:[/bold green]\n\n"
                    f"[blue]2D Solution:[/blue] {os.path.basename(file_2d)}\n"
                    f"[green]3D Solution:[/green] {os.path.basename(file_3d)}",
                    title="Comparison Setup",
                    border_style="cyan",
                )
            )

            # Load and validate data
            progress.update(
                main_task,
                description="[cyan]Loading and validating data...",
                advance=20,
            )

            # Perform comparison analysis
            progress.update(
                main_task,
                description="[cyan]Performing comparison analysis...",
                advance=30,
            )

            try:
                # Create comparison visualization
                progress.update(
                    main_task,
                    description="[cyan]Generating visualization...",
                    advance=20,
                )

                plot_comparison(file_2d, file_3d)

                # Final update
                progress.update(
                    main_task,
                    description="[green]Comparison completed successfully!",
                    completed=100,
                )

                # Display success message
                console.print(
                    Panel(
                        "[green]✓ Comparison analysis completed successfully![/green]\n\n"
                        "[blue]Summary:[/blue]\n"
                        f"• 2D Solution: {os.path.basename(file_2d)}\n"
                        f"• 3D Solution: {os.path.basename(file_3d)}\n"
                        "\n[yellow]The comparison plot has been generated.[/yellow]",
                        title="Analysis Complete",
                        border_style="green",
                    )
                )

            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Error during comparison analysis:[/red]\n{str(e)}\n\n"
                        "[yellow]Please check your data format and try again.[/yellow]",
                        title="Analysis Error",
                        border_style="red",
                    )
                )
                raise

    except KeyboardInterrupt:
        console.print(
            Panel(
                "[yellow]Process interrupted by user.[/yellow]",
                title="Cancelled",
                border_style="yellow",
            )
        )
        return
    except Exception as e:
        console.print(
            Panel(
                f"[red]Unexpected error:[/red]\n{str(e)}\n\n"
                "[yellow]Please report this issue if it persists.[/yellow]",
                title="Error",
                border_style="red",
            )
        )
        raise


def conservation_plots_menu():
    """Submenu for conservation plots functionality"""
    options = {
        "1": ("Energy Conservation Plot", "energy"),
        "2": ("Momentum Conservation Plot", "momentum"),
        "b": ("Back to Main Menu", None),
    }

    while True:
        print(term.clear)
        print_header()
        console.print(
            Panel(
                "[bold blue]Conservation Plots Menu:[/bold blue]",
                style="bold white",
                expand=False,
            )
        )

        for key, (desc, _) in options.items():
            print(term.cyan(f"[{key}] {desc}"))

        choice = input(term.bold("\nSelect an option: ")).strip().lower()

        if choice == "b":
            return

        if choice in options and options[choice][1]:
            plot_type = options[choice][1]
            try:
                # File selection mode
                console.print(
                    Panel(
                        "[bold blue]Select File Mode:[/bold blue]",
                        style="bold white",
                        expand=False,
                    )
                )
                file_mode = Prompt.ask(
                    "Select file mode", choices=["single", "multi"], default="single"
                )

                if file_mode == "multi":
                    chosen_csv = "multi_plot"
                    console.print("[bold]Select a folder containing CSV files:[/bold]")
                    selected_folder, _ = list_folders()
                    folder_path = os.path.join(".", selected_folder)
                else:
                    chosen_csv = list_items(root=".", select_type="file", file_keywords=["3D"])
                    if not chosen_csv:
                        console.print("[red]No suitable CSV files found.[/red]")
                        input(term.bold_red("\nPress Enter to continue..."))
                        continue

                # Legend style configuration
                use_method_legend = Confirm.ask(
                    "Use method names in legend instead of epsilon values?",
                    default=False,
                )

                # Update parameters
                parameters = CONFIG["DEFAULT_PARAMETERS"].copy()
                parameters.update(extract_parameters_by_file_name(chosen_csv))

                # Generate plot with progress bar
                with Progress() as progress:
                    task = progress.add_task("[cyan]Generating plot...", total=100)

                    try:
                        export_file_name = CONFIG["PLOT_TYPES"][plot_type][
                            "export_file_name"
                        ]
                        CONFIG["USE_METHOD_LEGEND"] = use_method_legend

                        plotter_conservation(
                            chosen_csv,
                            export_file_name,
                            parameters,
                            plot_type,
                            task
                        )
                        console.print('\n')
                        console.print(
                            Panel(
                                f"[green]✓ {options[choice][0]} generated successfully![/green]\n"
                                f"Output saved as: {export_file_name}",
                                title="Success",
                                border_style="green",
                            )
                        )

                    except Exception as e:
                        console.print('\n')
                        console.print(
                            Panel(
                                f"[red]Error during plot generation:[/red]\n{str(e)}",
                                title="Error",
                                border_style="red",
                            )
                        )
                        raise

                input(term.bold_green("\nPress Enter to continue..."))

            except Exception as e:
                console.print(f"[red]\nError: {str(e)}[/red]")
                input(term.bold_red("\nPress Enter to continue..."))
        else:
            console.print("[red]Invalid choice. Please try again.[/red]")
            input(term.bold_red("\nPress Enter to continue..."))


def main_menu():
    options = {
        "1": ("Package Management", check_and_install_requirements),
        "2": ("Peak Finder Analysis", peakfinder_menu),
        "3": ("2D Plotter", plotter_2d_menu),
        "4": ("3D Plotter", plotter_3d_menu),
        "5": ("Comparison Plots", plotter_comparison_menu),
        "6": ("Conservation Plots", conservation_plots_menu),
        "q": ("Quit", None),
    }

    while True:
        print(term.clear)
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
