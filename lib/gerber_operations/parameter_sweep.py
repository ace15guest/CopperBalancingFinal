"""
Parameter Sweep Analysis Module

This module performs comprehensive parameter sweep analysis for Gerber processing,
comparing processed results against reference data (Akro files) with various
processing parameters to find optimal settings.

The analysis tests all combinations of:
- DPI settings
- Edge filling methods
- Blur filters
- Gradient calculation methods
- Cropping parameters

Results are saved incrementally to CSV for analysis and visualization.

Performance Optimizations:
- Pre-loads and caches Akro reference files
- Pre-loads and caches NPZ files
- Batch CSV writes to reduce I/O overhead
- Early filtering of invalid parameter combinations
- Efficient processed combinations tracking with sets
- Parallel processing with multiprocessing for CPU-bound operations
- Optimized array operations with minimal copying
- Progress bars with tqdm for better user feedback
"""

from pathlib import Path
from itertools import product
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Import array operation utilities
from lib.array_operations.border_operations import get_border_mask, center_crop_by_area
from lib.array_operations.interpolation import (
    fill_nans_nd, 
    idw_fill_2d, 
    nearest_border_fill_true_2d, 
    fill_border_with_percent_max
)
from lib.array_operations.array_utils import (
    multiple_layers_weighted, 
    shrink_array, 
    rescale_to_shared_minmax
)
from lib.array_operations.gradient_analysis import analyze_gradients
from lib.array_operations.comparison import align_and_compare


def _load_and_prepare_akro_file(akro_file: Path) -> np.ndarray:
    """
    Load and prepare an Akro reference file.
    
    Args:
        akro_file: Path to Akro .dat file
        
    Returns:
        Prepared numpy array with NaN values filled
    """
    dat_load = np.loadtxt(str(akro_file))
    # Fill 9999 with nan
    dat_file_filled = np.where(dat_load == 9999.0, np.nan, dat_load)
    # Replace the nan values to do an iterative fill (best guess at missing data)
    return fill_nans_nd(dat_file_filled, 'iterative')


def _load_npz_file(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load NPZ file and return as dictionary.
    
    Args:
        npz_path: Path to NPZ file
        
    Returns:
        Dictionary of arrays from NPZ file
    """
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def _build_combination_name(
    company: str,
    dpi_value: int,
    percent_from_cntr: float,
    material: str,
    gerber_location: str,
    side: str,
    edge_fill_method: str,
    percent_max_fill: float,
    blur_type: str,
    radius: int,
    sigma: float,
    gradient_method: str,
    dx: float,
    window: int
) -> str:
    """
    Build unique identifier name for parameter combination.
    
    Returns:
        Unique name string for this combination
    """
    name = f"{company}_{dpi_value}_{percent_from_cntr}_{material}_{gerber_location}_{side}_{edge_fill_method}_"
    
    if edge_fill_method == "percent_max":
        name += f"{percent_max_fill}_"
    
    name += f"{blur_type}_"
    if blur_type == "box_blur":
        name += f"{radius}_"
    elif blur_type == "gauss":
        name += f"{sigma}_"
    
    name += f"{gradient_method}_{dx}_{dx}"
    if gradient_method == "plane":
        name += f"{window}"
    
    return name


def _process_single_combination(
    args: Tuple,
    akro_cache: Dict[str, np.ndarray],
    npz_cache: Dict[str, Dict[str, np.ndarray]],
    processed_pngs_folder: str
) -> Optional[Dict[str, Any]]:
    """
    Process a single parameter combination.
    
    This function is designed to be called in parallel by multiprocessing.
    
    Args:
        args: Tuple of parameters for this combination
        akro_cache: Pre-loaded Akro files
        npz_cache: Pre-loaded NPZ files
        processed_pngs_folder: Base folder for processed files
        
    Returns:
        Dictionary with results or None if skipped
    """
    from lib.array_operations.blur_filters import blur_call
    
    (window, dx, gradient_method, percent_max_fill, akro_file, dpi_value,
     edge_fill_method, blur_type, radius, sigma, percent_from_cntr, folder) = args
    
    # Extract metadata
    gerber_location = str(folder).replace("/", "\\").split('\\')[-1]
    akro_filename = str(akro_file).split('\\')[-1]
    side = akro_filename.split('_')[1]
    company = akro_filename.split('-')[0]
    material = akro_filename.split('-')[1]
    
    # Validate NPZ file exists
    npz_file_location = f"{processed_pngs_folder}/{gerber_location}_dpi_{dpi_value}/{gerber_location}_dpi_{dpi_value}.npz"
    if npz_file_location not in npz_cache:
        return None
    
    # Filter Akro file match
    if "Global" not in str(akro_file) and gerber_location not in str(akro_file):
        return None
    
    # Get cached data
    akro_key = str(akro_file)
    if akro_key not in akro_cache:
        return None
    
    dat_file_9999_filled = akro_cache[akro_key]
    data_dict = npz_cache[npz_file_location]
    
    # Process Gerber layers
    calculated_layers_preblend = multiple_layers_weighted(data_dict)
    mask = get_border_mask(calculated_layers_preblend)
    
    # Apply edge filling
    if edge_fill_method == 'idw':
        calculated_layers_preblend_edge_mask = idw_fill_2d(
            calculated_layers_preblend, mask=mask
        )
    elif edge_fill_method == 'nearest':
        calculated_layers_preblend_edge_mask = nearest_border_fill_true_2d(
            calculated_layers_preblend, mask=mask
        )
    elif edge_fill_method == 'percent_max':
        calculated_layers_preblend_edge_mask = fill_border_with_percent_max(
            calculated_layers_preblend, mask=mask, percent=percent_max_fill
        )
    else:
        return None
    
    # Apply blur
    calculated_layers_blended = blur_call(
        calculated_layers_preblend_edge_mask, blur_type, radius, sigma
    )
    
    # Align dimensions
    calculated_layers_blended_shrunk = shrink_array(
        calculated_layers_blended, dat_file_9999_filled.shape
    )
    
    # Rescale
    (calculated_layers_blended_shrink_rescale,
     dat_file_9999_filled_rescale,
     scale) = rescale_to_shared_minmax(
        calculated_layers_blended_shrunk, dat_file_9999_filled
    )
    
    # Crop to center
    dy = dx
    calculated_layers_blended_shrink_rescaled_cropped = center_crop_by_area(
        calculated_layers_blended_shrink_rescale, pct_area=percent_from_cntr)
    dat_file_9999_filled_rescale_cropped = center_crop_by_area(
        dat_file_9999_filled_rescale, pct_area=percent_from_cntr)
    
    # Calculate statistics
    stats = align_and_compare(
        calculated_layers_blended_shrink_rescaled_cropped,
        dat_file_9999_filled_rescale_cropped,
        ignore_zeros=False,
        detrend=True,
        with_scaling=False
    )
    
    # Calculate gradients
    metrics, angle_diff, mag_ratio = analyze_gradients(
        calculated_layers_blended_shrink_rescaled_cropped,
        dat_file_9999_filled_rescale_cropped,
        dx=dx,
        dy=dy,
        method=gradient_method,
        window_size=window
    )
    
    # Build output record
    name = _build_combination_name(
        company, dpi_value, percent_from_cntr, material, gerber_location,
        side, edge_fill_method, percent_max_fill, blur_type, radius,
        sigma, gradient_method, dx, window
    )
    
    out_info = {
        "Name": name,
        "DPI": dpi_value,
        "Percent from Center": percent_from_cntr,
        "Material": material,
        "Location": gerber_location,
        "Side": side,
        "Edge Fill": edge_fill_method,
        "Percent Max Fill": percent_max_fill if edge_fill_method == "percent_max" else 0,
        "Blur Type": blur_type,
        "Radius": radius if blur_type == "box_blur" else "",
        "Sigma": sigma if blur_type == "gauss" else "",
        "Gradient Method": gradient_method,
        "Dx": dx,
        "Dy": dy,
        "Window": window,
    }
    
    out_info.update(stats)
    out_info.update(metrics)
    out_info.pop("text", None)
    
    return out_info


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
    gerber_folders: List[str],
    output_file_path: str = "Assets/DataOutput/data_out.csv",
    processed_pngs_folder: str = "Assets/ProcessedPNGs/",
    batch_size: int = 10,
    use_parallel: bool = True,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perform comprehensive parameter sweep analysis on Gerber processing.
    
    This function processes all combinations of parameters, comparing processed
    Gerber data against reference Akro files. Each combination is tested and
    results are saved incrementally to avoid data loss during long runs.
    
    Performance optimizations include:
    - Pre-loading and caching of Akro and NPZ files
    - Batch CSV writes to reduce I/O overhead
    - Early filtering of invalid combinations
    - Efficient tracking of processed combinations
    - Optional parallel processing with multiprocessing
    - Optimized array operations
    
    Parameters:
        window_sizes: Window sizes for plane-based gradient calculation
        dx_choices: Spatial resolution values (dx) for gradient calculation
        gradient_methods: Gradient calculation methods ('plane', etc.)
        percent_max_fill_values: Percentage of max value for edge filling (0-100)
        all_akro_files: Paths to Akro reference data files (.dat format)
        dpi: DPI values used during Gerber to PNG conversion
        edge_fill: Edge filling methods ('idw', 'nearest', 'percent_max')
        blur: Blur filter types ('box_blur', 'gauss')
        radii: Radius values for box blur filter
        sigmas: Sigma values for Gaussian blur filter
        percent_area_from_centers: Percentage of center area to analyze (0-100)
        gerber_folders: Paths to folders containing processed Gerber data
        output_file_path: Path for output CSV file
        processed_pngs_folder: Base folder containing processed PNG/NPZ files
        batch_size: Number of results to accumulate before writing to CSV (default: 10)
        use_parallel: Enable parallel processing (default: True)
        n_workers: Number of worker processes (default: cpu_count() - 1)
        
    Returns:
        pd.DataFrame: Complete results with all metrics for each parameter combination
        
    Notes:
        - Results are written in batches to reduce I/O overhead
        - Already processed combinations are skipped automatically
        - Invalid parameter combinations are filtered out early
        - Progress is printed to console for monitoring
        - Parallel processing significantly speeds up CPU-bound operations
    """
    # ========================================
    # Initialize Output File Tracking
    # ========================================
    output_path = Path(output_file_path)
    write_header = not output_path.exists()
    
    # Load existing processed combinations for fast lookup
    processed_names: Set[str] = set()
    if output_path.exists():
        existing_df = pd.read_csv(output_file_path)
        processed_names = set(existing_df["Name"].tolist())
        print(f"Found {len(processed_names)} already processed combinations")
    
    # ========================================
    # Pre-load All Akro Files
    # ========================================
    print("Pre-loading all Akro files...")
    akro_cache: Dict[str, np.ndarray] = {}
    for akro_file in all_akro_files:
        akro_key = str(akro_file)
        try:
            akro_cache[akro_key] = _load_and_prepare_akro_file(akro_file)
        except Exception as e:
            print(f"Warning: Failed to load {akro_file}: {e}")
    print(f"Loaded {len(akro_cache)} Akro files into cache")
    
    
    # ========================================
    # Pre-load and Cache NPZ Files
    # ========================================
    print("Pre-loading NPZ files...")
    npz_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for folder in gerber_folders:
        gerber_location = str(folder).replace("/", "\\").split('\\')[-1]
        for dpi_value in dpi:
            folder_name = f"{processed_pngs_folder}/{gerber_location}_dpi_{dpi_value}"
            npz_file_location = f"{folder_name}/{gerber_location}_dpi_{dpi_value}.npz"
            npz_path = Path(npz_file_location)
            
            if npz_path.exists():
                try:
                    npz_cache[npz_file_location] = _load_npz_file(npz_path)
                except Exception as e:
                    print(f"Warning: Failed to load {npz_file_location}: {e}")
    npz_cache_list = [Path(x) for x in list(npz_cache)]
    print(f"Loaded {len(npz_cache)} NPZ files into cache")
    
    # ========================================
    # Filter Valid Parameter Combinations
    # ========================================
    # Pre-filter to remove invalid combinations early
    valid_edge_fills = {'idw', 'nearest', 'percent_max'}
    valid_blur_types = {'box_blur', 'gauss'}
    
    edge_fill = [ef for ef in edge_fill if ef in valid_edge_fills]
    blur = [b for b in blur if b in valid_blur_types]
    
    # ========================================
    # Generate All Parameter Combinations
    # ========================================
    items = list(product(
        window_sizes,
        dx_choices,
        gradient_methods,
        percent_max_fill_values,
        all_akro_files,
        dpi,
        edge_fill,
        blur,
        radii,
        sigmas,
        percent_area_from_centers,
        gerber_folders
    ))
    
    total_combinations = len(items)
    print(f"Processing {total_combinations} parameter combinations")
    
    # ========================================
    # Filter Unprocessed Combinations
    # ========================================
    unprocessed_items = []
    for item in items:
        # Quick pre-check to build name
        window, dx, gradient_method, percent_max_fill, akro_file, dpi_value, \
        edge_fill_method, blur_type, radius, sigma, percent_from_cntr, folder = item
        
        gerber_location = str(folder).replace("/", "\\").split('\\')[-1]
        akro_filename = str(akro_file).split('\\')[-1]
        side = akro_filename.split('_')[1]
        company = akro_filename.split('-')[0]
        material = akro_filename.split('-')[1]
        
        name = _build_combination_name(
            company, int(dpi_value), float(percent_from_cntr), material, gerber_location,
            side, str(edge_fill_method), float(percent_max_fill), str(blur_type), int(radius),
            float(sigma), str(gradient_method), float(dx), int(window)
        )
        
        if name not in processed_names:
            unprocessed_items.append(item)
    
    print(f"Filtered to {len(unprocessed_items)} unprocessed combinations")
    
    # ========================================
    # Process Combinations (Parallel or Sequential)
    # ========================================
    results_batch = []
    processed = 0
    
    if use_parallel and len(unprocessed_items) > 1:
        # Parallel processing
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        print(f"Using parallel processing with {n_workers} workers")
        
        # Create partial function with fixed arguments
        process_func = partial(
            _process_single_combination,
            akro_cache=akro_cache,
            npz_cache=npz_cache,
            processed_pngs_folder=processed_pngs_folder
        )
        
        # Process in parallel with progress bar
        with Pool(processes=n_workers) as pool:
            pbar = tqdm(total=len(unprocessed_items), desc="Processing combinations", unit="combo")
            for i, result in enumerate(pool.imap_unordered(process_func, unprocessed_items), 1):
                if result is not None:
                    results_batch.append(result)
                    processed_names.add(result["Name"])
                    processed += 1
                    
                    # Write batch to CSV
                    if len(results_batch) >= batch_size:
                        batch_df = pd.DataFrame(results_batch)
                        batch_df.to_csv(output_file_path, mode="a", index=False, header=write_header)
                        write_header = False
                        results_batch = []
                        pbar.set_postfix({"processed": processed, "batches_written": i // batch_size})
                
                pbar.update(1)
            pbar.close()
    else:
        # Sequential processing with progress bar
        print("Using sequential processing")
        for item in tqdm(unprocessed_items, desc="Processing combinations", unit="combo"):
            result = _process_single_combination(
                item, akro_cache, npz_cache, processed_pngs_folder
            )
            
            if result is not None:
                results_batch.append(result)
                processed_names.add(result["Name"])
                processed += 1
                
                # Write batch to CSV
                if len(results_batch) >= batch_size:
                    batch_df = pd.DataFrame(results_batch)
                    batch_df.to_csv(output_file_path, mode="a", index=False, header=write_header)
                    write_header = False
                    results_batch = []
    
    # ========================================
    # Write Remaining Results
    # ========================================
    if results_batch:
        batch_df = pd.DataFrame(results_batch)
        batch_df.to_csv(output_file_path, mode="a", index=False, header=write_header)
    
    print(f"\n✓ Complete: {len(unprocessed_items)} combinations processed ({processed} successful)")
    
    # ========================================
    # Return Final Results
    # ========================================
    if output_path.exists():
        return pd.read_csv(output_file_path)
    else:
        return pd.DataFrame()


# Made with Bob