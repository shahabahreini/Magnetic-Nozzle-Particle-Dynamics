import sys
import subprocess
import importlib.util
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


def get_terminal_width():
    """Get the terminal width, fallback to 80 if unable to determine"""
    return (
        shutil.get_terminal_size().columns
        if hasattr(shutil, "get_terminal_size")
        else 80
    )


def clear_screen():
    """Clear the console screen across different platforms"""
    os.system("cls" if os.name == "nt" else "clear")


def create_progress_bar(progress, total, width=30, title="Progress"):
    """Create a custom progress bar with title"""
    filled = int(width * progress // total)
    bar = "█" * filled + "░" * (width - filled)
    percentage = progress / total * 100
    return f"{title}: [{bar}] {percentage:0.1f}%"


def print_header():
    """Print an enhanced header"""
    clear_screen()
    width = get_terminal_width()
    print("╔" + "═" * (width - 2) + "╗")
    title = "Analysis Scripts Setup"
    padding = (width - len(title) - 2) // 2
    print("║" + " " * padding + title + " " * (width - len(title) - padding - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝\n")


def print_status(message, status="INFO", end="\n"):
    """Print a formatted status message with different status types"""
    status_colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m",  # Reset
    }

    try:
        color = status_colors.get(status.upper(), status_colors["INFO"])
        reset = status_colors["RESET"]
        print(f"{color}{message}{reset}", end=end)
    except:
        print(message, end=end)
    sys.stdout.flush()


def is_package_installed(package_name):
    """
    Improved package detection that handles common naming variations
    and checks both import names and pip names
    """
    # Strip version specifiers and extras
    base_name = (
        package_name.split("[")[0]
        .split("==")[0]
        .split(">=")[0]
        .split("<=")[0]
        .strip()
        .lower()
    )

    # Common package name mappings
    name_mappings = {
        "pyyaml": ["yaml"],
        "pyqt5": ["PyQt5"],
        # Add other mappings as needed
    }

    # Try direct import first
    if importlib.util.find_spec(base_name) is not None:
        return True

    # Try alternative names if available
    if base_name in name_mappings:
        for alt_name in name_mappings[base_name]:
            if importlib.util.find_spec(alt_name) is not None:
                return True

    # Try using pip to check if package is installed
    try:
        import pkg_resources

        pkg_resources.require(package_name)
        return True
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
        return False


def check_package(package_info):
    """Check if a package is installed using the improved detection"""
    package_key, package_spec = package_info
    return (package_key, package_spec, not is_package_installed(package_key))


def read_requirements():
    """Read requirements from requirements.txt file with improved package name handling"""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print_status("Error: requirements.txt file not found!", "ERROR")
        return {}

    requirements = {}
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle package names more carefully
                package = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                # Store original spec but use normalized key
                requirements[package.lower()] = line
    return requirements


def install_packages_parallel(missing_packages):
    """Install packages in parallel with progress tracking"""
    total = len(missing_packages)
    completed = 0

    print_status("\nInstalling missing packages...")
    print(create_progress_bar(0, total, title="Installing packages"))

    with ThreadPoolExecutor(max_workers=min(total, 4)) as executor:
        futures = {
            executor.submit(install_package, package): package
            for package in missing_packages
        }

        for future in as_completed(futures):
            package = futures[future]
            completed += 1
            success = future.result()

            # Clear the previous line
            print("\033[F\033[K", end="")
            print(create_progress_bar(completed, total, title="Installing packages"))


def install_package(package):
    """Install a package using pip with better error handling and output"""
    try:
        # First try to import to double check if it's really missing
        package_base = package.split("==")[0].split(">=")[0].split("<=")[0].strip()

        # Special handling for known packages
        if package_base.lower() == "pyyaml":
            try:
                import yaml

                return True
            except ImportError:
                pass
        elif package_base.lower() == "pyqt5":
            try:
                import PyQt5

                return True
            except ImportError:
                pass

        # If not already installed, try to install
        print_status(f"\nInstalling {package}...", "INFO")
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            capture_output=True,
            text=True,
        )

        if process.returncode != 0:
            print_status(f"\nError installing {package}: {process.stderr}", "ERROR")

            # Try alternative installation methods for problematic packages
            if package_base.lower() == "pyqt5":
                print_status("\nTrying alternative PyQt5 installation...", "INFO")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "PyQt5-sip"]
                )
                process = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "PyQt5"],
                    capture_output=True,
                    text=True,
                )

            if process.returncode != 0:
                return False

        return True
    except Exception as e:
        print_status(f"\nException while installing {package}: {str(e)}", "ERROR")
        return False


def check_and_install_requirements():
    """Check and install required packages from requirements.txt with improved error handling"""
    print_header()
    print_status("Checking package dependencies...\n")

    required_packages = read_requirements()
    if not required_packages:
        return False

    # Check packages
    missing_packages = []
    total_packages = len(required_packages)
    completed = 0

    print(create_progress_bar(0, total_packages, title="Checking packages"))

    with ThreadPoolExecutor(max_workers=min(total_packages, 8)) as executor:
        futures = [
            executor.submit(check_package, item) for item in required_packages.items()
        ]

        for future in as_completed(futures):
            completed += 1
            package_key, package_spec, is_missing = future.result()
            if is_missing:
                missing_packages.append(package_spec)

            print("\033[F\033[K", end="")
            print(
                create_progress_bar(
                    completed, total_packages, title="Checking packages"
                )
            )

    if missing_packages:
        # Try to install packages one by one for better error handling
        print_status("\nInstalling missing packages...")
        failed_packages = []

        for i, package in enumerate(missing_packages, 1):
            success = install_package(package)
            if not success:
                failed_packages.append(package)

            # Update progress bar
            print("\033[F\033[K", end="")
            print(
                create_progress_bar(
                    i, len(missing_packages), title="Installing packages"
                )
            )

        if failed_packages:
            print_status(
                f"\nWarning: Failed to install: {', '.join(failed_packages)}", "WARNING"
            )
            print_status("\nTry installing these packages manually using:")
            for package in failed_packages:
                print(f"pip install {package}")
    else:
        print_status("\nAll packages are already installed!", "SUCCESS")

    return True


def main():
    """Main function to run the application"""
    try:
        if check_and_install_requirements():
            try:
                print_status("\nStarting main plotter application...")
                from main_plotter import main_menu

                main_menu()
            except ImportError as e:
                print_status(
                    "\nError: main_plotter.py not found or contains errors.", "ERROR"
                )
                return False
        return True
    except KeyboardInterrupt:
        print_status("\n\nProcess interrupted by user.", "WARNING")
        return False
    except Exception as e:
        print_status(f"\nError: {str(e)}", "ERROR")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print()
        input("Press Enter to exit...")
