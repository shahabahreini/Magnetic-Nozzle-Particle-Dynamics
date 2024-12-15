import sys
import subprocess
import importlib.util
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def clear_screen():
    """Clear the console screen across different platforms"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a simple header"""
    clear_screen()
    print("=" * 50)
    print("Analysis Scripts Setup")
    print("=" * 50)
    print()

def print_status(message, end="\n"):
    """Print a status message with a timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", end=end)
    sys.stdout.flush()

def is_package_installed(package_name):
    """Check if a package is installed using importlib (faster than importing)"""
    try:
        package_base_name = package_name.split("[")[0].split("==")[0].split(">=")[0].split("<=")[0].strip().lower()
        return importlib.util.find_spec(package_base_name) is not None
    except Exception:
        return False

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"\nWarning: Error installing {package}: {str(e)}")
        return False

def read_requirements():
    """Read requirements from requirements.txt file"""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        return {}

    requirements = {}
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                package = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                requirements[package.lower()] = line
    return requirements

def check_package(package_info):
    """Check a single package (for parallel processing)"""
    package_key, package_spec = package_info
    return (package_key, package_spec, not is_package_installed(package_key))

def install_packages_parallel(missing_packages):
    """Install packages in parallel"""
    with ThreadPoolExecutor(max_workers=min(len(missing_packages), 4)) as executor:
        futures = {executor.submit(install_package, package): package for package in missing_packages}
        for future in as_completed(futures):
            package = futures[future]
            try:
                success = future.result()
                if success:
                    print(f"Successfully installed {package}")
                else:
                    print(f"Failed to install {package}")
            except Exception as e:
                print(f"Error installing {package}: {str(e)}")

def check_and_install_requirements():
    """Check and install required packages from requirements.txt"""
    print_header()
    print_status("Checking package dependencies...")
    
    # Quick check for pip
    if not is_package_installed("pip"):
        print("Error: pip is not installed. Please install pip first.")
        return False

    required_packages = read_requirements()
    if not required_packages:
        print("Error: No requirements found. Please check requirements.txt file.")
        return False

    # Check packages in parallel
    missing_packages = []
    with ThreadPoolExecutor(max_workers=min(len(required_packages), 8)) as executor:
        futures = [executor.submit(check_package, item) for item in required_packages.items()]
        for future in as_completed(futures):
            try:
                package_key, package_spec, is_missing = future.result()
                if is_missing:
                    missing_packages.append(package_spec)
            except Exception as e:
                print(f"Error checking package: {str(e)}")

    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        install_packages_parallel(missing_packages)
    else:
        print("\nAll packages are already installed!")

    # Quick final verification
    print("\nVerifying installations...")
    failed = []
    for package_key in required_packages:
        if not is_package_installed(package_key):
            failed.append(package_key)

    if failed:
        print(f"\nWarning: Some packages may not be properly installed: {', '.join(failed)}")
    else:
        print("\nAll packages verified successfully!")
    
    return True

def main():
    """Main function to run the application"""
    try:
        if check_and_install_requirements():
            try:
                print("\nStarting main plotter application...")
                from main_plotter import main_menu
                main_menu()
            except ImportError as e:
                print(f"\nError importing main_plotter.py: {str(e)}")
                print("\nPlease ensure main_plotter.py is in the same directory.")
                return False
        return True
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)
