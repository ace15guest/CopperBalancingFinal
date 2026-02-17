"""
Main entry point for the Copper Balancing Image Processing application.

This module provides the main execution logic and CLI interface.
"""

from lib.gerber_operations.parameter_sweep import parameter_sweep_analysis
from src.config import Config


def main():
    """Main entry point for the application."""
    print("Copper Balancing Image Processing")
    print("Version 0.1.0")
    # TODO: Implement main application logic
    window_sizes = Config.PARAMETER_SWEEPS["window_sizes"]
    dx_choices = Config.PARAMETER_SWEEPS["dx_choices"]
    gradient_methods = Config.PARAMETER_SWEEPS["gradient_methods"]
    percent_max_fill_values = Config.PARAMETER_SWEEPS["percent_max_fills"]
    all_akro_files = Config.ALL_AKRO_FILES
    dpi = Config.PARAMETER_SWEEPS["dpis"]
    edge_fill = Config.PARAMETER_SWEEPS["edge_fills"]
    blur = Config.PARAMETER_SWEEPS["blur_types"]
    radii = Config.PARAMETER_SWEEPS["radii"]
    sigmas = Config.PARAMETER_SWEEPS["sigmas"]
    percent_area_from_centers = Config.PARAMETER_SWEEPS["percent_area_from_centers"]
    gerber_folders = Config.get_gerber_dirs()
    output_path = Config.DATA_OUTPUT_FILE

    parameter_sweep_analysis(
        window_sizes=window_sizes,
        dx_choices=dx_choices,
        gradient_methods=gradient_methods,
        percent_max_fill_values=percent_max_fill_values,
        all_akro_files=all_akro_files,
        dpi=dpi,
        edge_fill=edge_fill,
        blur=blur,
        radii=radii,
        sigmas=sigmas,
        percent_area_from_centers=percent_area_from_centers,
        gerber_folders=gerber_folders,
        output_file_path=str(output_path),
        processed_pngs_folder = str(Config.PROCESSED_PNG_FOLDER)
    )



if __name__ == "__main__":
    main()

# Made with Bob
