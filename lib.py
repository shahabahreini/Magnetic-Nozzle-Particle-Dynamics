import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def search_for_export_csv():
    # get all files in current directory
    files = os.listdir()

    # filter out files with csv extension
    csv_files = [file for file in files if file.endswith(".csv")]

    # print the list of csv files
    print("CSV files in current directory:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")

    # ask user to choose a file
    while True:
        choice = input("Choose a file (enter a number from the list): ")
        try:
            choice = int(choice)
            if choice > 0 and choice <= len(csv_files):
                break
        except ValueError:
            pass
        print("Invalid choice, please try again.")

    # print the selected file name
    selected_file = csv_files[choice - 1]
    print(f"Selected file: {selected_file}")

    return selected_file


def extract_parameters_by_file_name(fname):
    numbers = {}

    # Adjusted regex pattern to handle scientific notation (e.g., 1.23e-8)
    pattern = r"(eps|epsphi|kappa|deltas|beta|alpha|theta|time)(\d+\.\d+(?:e[+-]?\d+)?)"

    for match in re.finditer(pattern, fname):
        key = match.group(1)
        value = float(match.group(2))  # Converts string directly to float, handling scientific notation
        numbers[key] = value

    return numbers


def read_exported_csv_simulation(path_, fname_):
    """Gets the folder path and desired file name and load the data into Pandas DataFrame"""

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


def read_exported_csv_2Dsimulation(path_, fname_):
    """Gets the folder path and desired file name and load the data into Pandas DataFrame"""
    fpath = os.path.join(path_, fname_)
    data = pd.read_csv(fpath)
    df = pd.DataFrame(
        data, columns=["timestamp", "omega_rho",
                       "omega_z", "rho", "z", "drho", "dz", "dphi"]
    )

    return df


def adiabtic_calculator(v_x, x, extremum_idx, label=None):
    velocity = v_x
    position = x

    # Compute the changes in the components of X
    delta_X = position.diff()

    # Compute the cumulative sum of velocity times delta_X
    adiabatic = np.cumsum(velocity * delta_X)

    # Compute the integral of V.dX between sequential extremum indexes
    integral_VdX = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        integral = np.sum(velocity[start_idx:end_idx]
                          * delta_X[start_idx:end_idx])
        integral_VdX.append(integral)

    # Plot the integral versus cycles
    # plt.plot(range(len(integral_VdX)), integral_VdX)
    # plt.xlabel('Cycles')
    # plt.ylabel(r'$\oint\, V.\, dX$')
    # plt.title('Closed Path Integral Of Radial Velocity per Cycles')
    # plt.show()

    return integral_VdX


def adiabtic_calculator_fixed(v_x, x, extremum_idx, label=None):
    velocity = v_x
    position = x

    # Compute the changes in the components of X
    delta_X = position.diff()

    # Compute the cumulative sum of velocity times delta_X
    adiabatic = np.cumsum(velocity * delta_X)

    # Compute the integral of V.dX between sequential extremum indexes
    integral_VdX = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        integral = np.sum(velocity[start_idx:end_idx]
                          * delta_X[start_idx:end_idx])
        integral_VdX.append(integral)

    # Plot the integral versus cycles
    plt.plot(range(len(integral_VdX)), integral_VdX, label=label)
    plt.xlabel('Cycles')
    plt.ylabel(r'$\oint\, V.\, dX$')
    plt.title('Closed Path Integral Of Radial Velocity per Cycles')

    return adiabatic


def magnetic_change_calculate(B_x, B_z, extremum_idx, label=None):
    # Calculate the magnitude of the magnetic field vector at each point
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    # Compute the relative changes in the magnitude of the magnetic field
    relative_magnetic_changes = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        initial_magnitude = B_magnitude[start_idx]
        final_magnitude = B_magnitude[end_idx - 1]
        relative_change = (final_magnitude -
                           initial_magnitude) / initial_magnitude * 100
        relative_magnetic_changes.append(relative_change)

    # Plot the relative magnetic changes versus cycles
    plt.plot(range(len(relative_magnetic_changes)),
             relative_magnetic_changes, label=label)
    plt.axhline(y=-0.065, color='r', linestyle='--',
                label='Threshold (0.2)')  # Add a dashed line at 0.2
    plt.xlabel('Cycles')
    plt.ylabel(r'Relative $\Delta B$ (%)')
    plt.title('Relative Magnetic Field Changes per Cycles')
    plt.legend()

    return relative_magnetic_changes


def epsilon_calculate(B_x, B_z, extremum_idx, time, label=None):
    # Calculate the magnitude of the magnetic field vector at each point
    """
    The epsilon_calculate function calculates the dimensionless parameter epsilon for each cycle.

    :param B_x: Calculate the magnitude of the magnetic field vector at each point
    :param B_z: Calculate the magnitude of the magnetic field vector
    :param extremum_idx: Find the indices of the local maxima and minima in b_magnitude
    :param time: Calculate the time duration of one gyration cycle
    :param label: Label the plot
    :return: A list of epsilon values for each cycle
    :doc-author: Trelent
    """
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    # Compute epsilon for each cycle
    start_idx = extremum_idx[0]
    end_idx = extremum_idx[1]
    omega_g = B_magnitude[start_idx]
    #omega_g = ((1-B_magnitude[end_idx]/B_magnitude[start_idx])**2)*B_magnitude[start_idx]# Assuming omega_g is proportional to B
    # Time duration of one gyration cycle
    tau_B = time[end_idx] - time[start_idx]
    epsilon_i = omega_g * tau_B

    epsilon_values = []
    for i in range(len(extremum_idx) - 1):
        start_idx = extremum_idx[i]
        end_idx = extremum_idx[i + 1]
        omega_g = B_magnitude[start_idx]  # Assuming omega_g is proportional to B
        # omega_g = (B_magnitude[start_idx]**2 / (B_magnitude[end_idx])
        # Time duration of one gyration cycle
        tau_B = time[end_idx] - time[start_idx]
        epsilon = omega_g * tau_B / epsilon_i
        epsilon_values.append(epsilon)

    # Plot epsilon versus cycles
    plt.plot(range(len(epsilon_values)), epsilon_values, label=label)
    plt.xlabel('Cycles')
    plt.ylabel(r'$\epsilon$')
    plt.title(r'Dimensionless Parameter $\epsilon$ per Cycles')
    plt.legend()

    return epsilon_values


def calculate_dynamic_epsilon(data, q=1, m=1, label=None):

    # Calculate velocity components
    data['v_rho'] = data['drho']
    data['v_phi'] = data['rho'] * data['dphi']
    data['v_z'] = data['dz']

    # Calculate the magnitude of the magnetic field
    data['B'] = np.sqrt(data['Magnetic_rho']**2 + data['Magnetic_z']**2)

    # Calculate the gradient of the magnetic field using finite differences
    data['grad_B_rho'] = data['B'].diff() / data['rho'].diff()
    data['grad_B_z'] = data['B'].diff() / data['z'].diff()
    data['grad_B_rho'].fillna(0, inplace=True)  # Handle NaN values
    data['grad_B_z'].fillna(0, inplace=True)

    # Calculate the dot product of grad_B and velocity
    data['grad_B_dot_v'] = data['grad_B_rho'] * \
        data['v_rho'] + data['grad_B_z'] * data['v_z']

    # Calculate epsilon
    data['epsilon'] = (q / m) * (data['B']**2) / data['grad_B_dot_v']

    plt.plot(range(len(data['epsilon'])), data['epsilon'], label=label)
    plt.xlabel('Cycles')
    plt.ylabel(r'$\epsilon$')
    plt.title(r'Dimensionless Parameter $\epsilon$ per Cycles')
    plt.legend()

    return data[['timestamp', 'epsilon']]



def epsilon_calculate_allPoints(B_x, B_z, time, label=None):
    # Calculate the magnitude of the magnetic field vector at each point
    """
    The epsilon_calculate function calculates the dimensionless parameter epsilon for each cycle.

    :param B_x: Calculate the magnitude of the magnetic field vector at each point
    :param B_z: Calculate the magnitude of the magnetic field vector
    :param extremum_idx: Find the indices of the local maxima and minima in b_magnitude
    :param time: Calculate the time duration of one gyration cycle
    :param label: Label the plot
    :return: A list of epsilon values for each cycle
    :doc-author: Trelent
    """
    B_magnitude = np.sqrt(B_x**2 + B_z**2)

    # Compute epsilon for each cycle
    omega_g = B_magnitude[0]  # Assuming omega_g is proportional to B
    # Time duration of one gyration cycle
    tau_B = time[1] - time[0]
    epsilon_i = omega_g * tau_B

    epsilon_values = []
    for i in range(len(time) - 1):
        omega_g = (B_magnitude[i]**2 / B_magnitude[i+1])
        # Time duration of one gyration cycle
        tau_B = time[i+1] - time[i]
        epsilon = omega_g * tau_B / epsilon_i
        epsilon_values.append(epsilon)

    # Plot epsilon versus cycles
    plt.plot(range(len(epsilon_values)), epsilon_values, label=label)
    plt.xlabel('Cycles')
    plt.ylabel(r'$\epsilon$')
    plt.title(r'Dimensionless Parameter $\epsilon$ per Cycles')
    plt.legend()
    

def calculate_magnetic_field(rho, z):
    """
    Calculate magnetic field components in cylindrical coordinates.
    """
    B_r = rho / (rho**2 + z**2)**(3/2)
    B_z = z / (rho**2 + z**2)**(3/2)
    B_phi = 0
    return B_r, B_phi, B_z

def calculate_guiding_center(B, v, rho, z):
    """
    Calculate the guiding center correction.
    """
    B_mag_sq = np.dot(B, B)
    v_cross_B = np.cross(v, B)
    R_gc = v_cross_B / B_mag_sq
    r_gc_rho = rho - R_gc[0]
    r_gc_z = z - R_gc[2]
    return r_gc_rho, r_gc_z

def calculate_ad_mio(df, label=None, use_guiding_center=True):
    """
    The calculate_ad_mio function calculates the adiabatic invariant mu (magnetic moment) 
    for a charged particle in a magnetic field at each point in the DataFrame.
    """
    # Constants
    m = 1  # Mass of the particle (adjust as needed)
    
    df['v_rho'] = df['drho']
    df['v_phi'] = df['rho'] * df['dphi']
    df['v_z'] = df['dz']
    
    mu_values = []
    
    for i, row in df.iterrows():
        # Compute magnetic field components at the particle's position
        B_r, B_phi, B_z = calculate_magnetic_field(row['rho'], row['z'])
        B = np.array([B_r, B_phi, B_z])
        v = np.array([row['v_rho'], row['v_phi'], row['v_z']])
        
        if use_guiding_center:
            # Compute magnetic field components at the guiding center
            r_gc_rho, r_gc_z = calculate_guiding_center(B, v, row['rho'], row['z'])
            B_r, B_phi, B_z = calculate_magnetic_field(r_gc_rho, r_gc_z)
            B = np.array([B_r, B_phi, B_z])
        
        # Calculate the magnitude of the magnetic field
        B_magnitude = np.linalg.norm(B)
        
        # Compute perpendicular velocity component
        B_unit = B / B_magnitude
        v_perp_vector = v - np.dot(v, B_unit) * B_unit
        v_perp_magnitude = np.linalg.norm(v_perp_vector)
        
        # Compute mu for each point
        mu = m * v_perp_magnitude**2 / (2 * B_magnitude)
        mu_values.append(mu)

    # Plot mu versus time points
    plt.plot(df['timestamp'], mu_values, label=label)
    plt.xlabel('Time')
    plt.ylabel(r'$\mu$')
    if use_guiding_center:
        plt.title(r'Adiabatic Invariant $\mu = \frac{m v_{\perp}^2}{2 B}$ per Time Points'+'\nCalculated Based on Magnetic Field at Guiding Center')
    else:
        plt.title(r'Adiabatic Invariant $\mu = \frac{m v_{\perp}^2}{2 B}$ per Time Points'+'\nCalculated Based on Magnetic Field at Particle Position')
    plt.legend()
    
    return mu_values
