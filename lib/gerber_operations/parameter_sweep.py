"""
Parameter Sweep Analysis

This module handles comprehensive parameter sweep analysis for Gerber processing,
comparing processed results against reference data (Akro files) with various
processing parameters.

TODO: Optimize this function - currently a direct paste for initial implementation.
"""
from pathlib import Path
from itertools import product
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from array_operations.array_utils import fill_nans_nd, get_border_mask, idw_fill_2d, nearest_border_fill_true_2d, fill_border_with_percent_max
from array_operations.array_utils import multiple_layers_weighted


def parameter_sweep_analysis(
    window_sizes: List[int],
    dx_choices: List[float],
    gradient_methods: List[str],
    percent_max_fill_values: List[float],
    all_akro_files: List[Path],
    dpi: List[int],
    edge_fill: List[str],
    blur: List[str],
    radii: List[int],
    sigmas: List[float],
    percent_area_from_centers: List[float],
    gerber_folders: List[Path],
    output_file_path: str = "Assets/DataOutput/data_out.csv"
) -> pd.DataFrame:
    """
    Perform comprehensive parameter sweep analysis on Gerber processing.
    
    This function processes all combinations of parameters, comparing processed
    Gerber data against reference Akro files with various processing settings.
    
    Parameters:
        window_sizes: List of window sizes for gradient calculation
        dx_choices: List of dx values for gradient calculation
        gradient_methods: List of gradient methods ('plane', etc.)
        percent_max_fill_values: List of percent max fill values for edge filling
        all_akro_files: List of paths to Akro reference data files
        dpi: List of DPI values to process
        edge_fill: List of edge fill methods ('idw', 'nearest', 'percent_max')
        blur: List of blur types ('box_blur', 'gauss')
        radii: List of radius values for box blur
        sigmas: List of sigma values for Gaussian blur
        percent_area_from_centers: List of percent area from center for cropping
        gerber_folders: List of Gerber folder paths to process
        output_file_path: Output CSV file path (default: "Assets/DataOutput/data_out.csv")
        
    Returns:
        pd.DataFrame: Results dataframe with all metrics
        
    Note:
        This function needs optimization. It's a direct paste of working code
        for initial implementation. Future improvements should include:
        - Better error handling
        - Progress tracking
        - Parallel processing
        - Memory optimization
        - Modular refactoring
    """
    # Import required functions here (adjust based on your actual module structure)
    from lib.array_operations.blur_filters import blur_call
    # Add other imports as needed
    
    # Initialize output tracking
    write_header = True
    count = 0
    
    # Create all combinations
    items = list(product(
        window_sizes, dx_choices, gradient_methods, percent_max_fill_values,
        all_akro_files, dpi, edge_fill, blur, radii, sigmas,
        percent_area_from_centers, gerber_folders
    ))
    
    # Process all combinations of parameters
    for window, dx, gradient_method, percent_max_fill, akro_file, dpi_value, edge_fill_method, blur_type, radius, sigma, percent_from_cntr, folder in items:
        count += 1
        print(f"Processing combination {count}/{len(items)}")
        
        info = [folder, window, dx, gradient_method, percent_max_fill, akro_file, 
                dpi_value, edge_fill_method, blur_type, radius, sigma, percent_from_cntr]
        
        # Extract metadata from paths
        gerber_location = str(folder).split('\\')[-1]
        side = str(akro_file).split('\\')[-1].split('_')[1]
        company = str(akro_file).split('\\')[-1].split('-')[0]
        material = str(akro_file).split('\\')[-1].split('-')[1]
        
        # Build output name
        name = f"{company}_{dpi_value}_{percent_from_cntr}_{material}_{gerber_location}_{side}_{edge_fill_method}_"
        
        if edge_fill_method == "percent_max":
            name += f"{percent_max_fill}_"
        elif edge_fill_method == "nearest":
            percent_max_fill = ""
        elif edge_fill_method == "idw":
            percent_max_fill = ""
        else:
            print("Skipped - invalid edge fill method")
            continue
        
        name += f"{blur_type}_"
        if blur_type == "box_blur":
            name += f"{radius}_"
            sigma = ""
        elif blur_type == "gauss":
            name += f"{sigma}_"
            radius = ""
        else:
            print("Skipped - invalid blur type")
            continue
        
        name += f"{gradient_method}_{int(dx)}_{int(dx)}"
        if gradient_method == "plane":
            name += f"{window}"
        
        # Locate NPZ file
        folder_name = Path(f"Assets/processed_pngs/{gerber_location}_dpi_{dpi_value}").expanduser().resolve()
        npz_file_location = f"{folder_name}/{gerber_location}_dpi_{dpi_value}.npz"
        
        # Check if NPZ exists
        if not Path(npz_file_location).exists():
            print(f"NPZ file not found at {npz_file_location}, skipping processing.")
            continue
        
        # Check if already processed
        if Path(output_file_path).exists():
            existing_df = pd.read_csv(output_file_path)
            if name in list(existing_df["Name"]):
                print(f"Combination {name} already processed, skipping.")
                continue
        
        # Filter for Global files matching location
        if "Global" not in str(akro_file) or gerber_location not in str(akro_file):
            continue
        else:
            dat_load = np.loadtxt(str(akro_file))
            dat_file_filled = np.where(dat_load == 9999.0, np.nan, dat_load)
            dat_file_9999_filled = fill_nans_nd(dat_file_filled, 'iterative')
        
        # Load NPZ data
        with np.load(npz_file_location) as data:
            data_dict = {key: data[key] for key in data.files}
        
        # Process layers
        calculated_layers_preblend = multiple_layers_weighted(data_dict)
        mask = get_border_mask(calculated_layers_preblend)
        
        # Apply edge filling
        if edge_fill_method == 'idw':
            calculated_layers_preblend_edge_mask = idw_fill_2d(calculated_layers_preblend, mask=mask)
        elif edge_fill_method == 'nearest':
            calculated_layers_preblend_edge_mask = nearest_border_fill_true_2d(calculated_layers_preblend, mask=mask)
        elif edge_fill_method == 'percent_max':
            calculated_layers_preblend_edge_mask = fill_border_with_percent_max(
                calculated_layers_preblend, mask=mask, percent=int(percent_max_fill)
            )
        
        # Apply blur
        calculated_layers_blended = blur_call(calculated_layers_preblend_edge_mask, blur_type, radius, sigma)
        
        # Shrink or grow the array to match the larger array
        calculated_layers_blended_shrunk = shrink_array(calculated_layers_blended, dat_file_9999_filled.shape)
        
        # Rescale to share the same min and max
        calculated_layers_blended_shrink_rescale, dat_file_9999_filled_rescale, scale = rescale_to_shared_minmax(
            calculated_layers_blended_shrunk, dat_file_9999_filled
        )
        
        # Crop to center area
        dy = dx
        calculated_layers_blended_shrink_rescaled_cropped = center_crop_by_area(
            calculated_layers_blended_shrink_rescale, pct_area=percent_from_cntr
        )
        dat_file_9999_filled_rescale_cropped = center_crop_by_area(
            dat_file_9999_filled_rescale, pct_area=percent_from_cntr
        )
        
        # Calculate statistics
        stats = align_and_compare(
            calculated_layers_blended_shrink_rescaled_cropped,
            dat_file_9999_filled_rescale_cropped,
            ignore_zeros=False,
            detrend=True,
            with_scaling=False
        )
        
        metrics, angle_diff, mag_ratio = analyze_gradients(
            calculated_layers_blended_shrink_rescale,
            dat_file_9999_filled_rescale,
            dx=dx,
            dy=dy,
            method=gradient_method,
            window_size=window,
            make_plots=False
        )
        
        # Build output info
        out_info = {
            "Name": name,
            "DPI": dpi_value,
            "Percent from Center": percent_from_cntr,
            "Material": material,
            "Location": gerber_location,
            "Side": side,
            "Edge Fill": edge_fill_method,
            "Percent Max Fill": percent_max_fill,
            "Blur Type": blur_type,
            "Radius": radius,
            "Sigma": sigma,
            "Gradient Method": gradient_method,
            "Dx": dx,
            "Dy": dy,
            "Window": window,
        }
        out_info.update(stats)
        out_info.update(metrics)
        out_info.pop("text", None)  # Remove text field if exists
        
        # Save to CSV
        out_df = pd.DataFrame([out_info])
        out_df.to_csv(output_file_path, mode="a", index=False, header=write_header)
        
        # Cleanup
        del out_df
        del stats
        write_header = False
        
        print(f"Processing with DPI: {dpi_value}, Edge Fill: {edge_fill_method}, Blur: {blur_type}")
    
    # Return final results
    if Path(output_file_path).exists():
        return pd.read_csv(output_file_path)
    else:
        return pd.DataFrame()


# Made with Bob