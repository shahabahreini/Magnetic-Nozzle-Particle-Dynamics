from .config import Configuration
from .file_utils import (
    print_styled,
    search_for_export_csv,
    extract_parameters_by_file_name,
    read_exported_csv_simulation,
    read_exported_csv_simulatio_3D,
    read_exported_csv_2Dsimulation,
    list_folders,
    list_csv_files,
    list_csv_files_noFolder,
    find_common_and_varying_params,  # Added this line
)
from .magnetic_field import (
    B_rho,
    B_z,
    B_phi,
    B_magnitude,
    gradient_B_magnitude,
    L_B,
    calculate_magnetic_field,
)
from .adiabatic import (
    adiabtic_calculator,
    adiabatic_calculator_noCycles,
    adiabtic_calculator_fixed,
)
from .physics import (
    calculate_velocity_components,
    calculate_guiding_center,
    calculate_adiabaticity,
    magnetic_change_calculate,
)
from .visualization import (
    get_axis_label,
    save_plots_with_timestamp,
    calculate_ad_mio,
)
from .epsilon import (
    epsilon_calculate,
    epsilon_calculate_allPoints,
    calculate_dynamic_epsilon,
)

__all__ = [
    "Configuration",
    "print_styled",
    "search_for_export_csv",
    "extract_parameters_by_file_name",
    "read_exported_csv_simulation",
    "read_exported_csv_simulatio_3D",
    "read_exported_csv_2Dsimulation",
    "list_folders",
    "list_csv_files",
    "list_csv_files_noFolder",
    "find_common_and_varying_params",  # Added this line
    "B_rho",
    "B_z",
    "B_phi",
    "B_magnitude",
    "gradient_B_magnitude",
    "L_B",
    "calculate_magnetic_field",
    "adiabtic_calculator",
    "adiabatic_calculator_noCycles",
    "adiabtic_calculator_fixed",
    "calculate_velocity_components",
    "calculate_guiding_center",
    "calculate_adiabaticity",
    "magnetic_change_calculate",
    "get_axis_label",
    "save_plots_with_timestamp",
    "calculate_ad_mio",
    "epsilon_calculate",
    "epsilon_calculate_allPoints",
    "calculate_dynamic_epsilon",
]
