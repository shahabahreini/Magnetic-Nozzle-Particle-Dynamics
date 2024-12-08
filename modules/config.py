import yaml


class Configuration:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)

        # File naming and format
        self.save_file_name = self.config["save_file_name"]
        self.save_file_extension = self.config["save_file_extension"]

        # File handling settings
        self.is_multi_files = self.config["is_multi_files"]
        self.target_folder = self.config["target_folder_multi_files"]
        self.plots_folder = self.config["plots_folder"]

        # Analysis settings
        self.extremum_of = self.config["extremum_of"]
        self.based_on_guiding_center = self.config["based_on_guiding_center"]
        self.calculate_integral = self.config["calculate_integral"]
        self.calculate_traditional_magneticMoment = self.config[
            "calculate_traditional_magneticMoment"
        ]
        self.show_extremums_peaks = self.config["show_extremums_peaks"]
        self.show_amplitude_analysis = self.config["show_amplitude_analysis"]

        # Plot settings
        self.share_x_axis = self.config["SHARE_X_AXIS"]

        # Simulation parameters
        self.simulation_time = self.config["simulation_time"]
        self.method = self.config["method"]

        # Optional: full parameter dictionary
        self.parameter_dict = self.config.get("simulation_parameters", {})

    def load_config(self, config_path):
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)

    def __str__(self):
        """Pretty print configuration"""
        return yaml.dump(self.config, default_flow_style=False)
