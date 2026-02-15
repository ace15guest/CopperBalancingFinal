"""
Batch Processor

This module handles batch processing of Gerber files and layers.
"""

from pathlib import Path
from conversion_operations.gerber_converter import wait_for_file_stability
from typing import List, Dict, Callable, Optional
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import numpy as np
import glob
import os


def batch_gerber_to_png_multi_dpi(gerber_folders: List[Path], dpi_list: List[int], output_base_dir: str = "Assets/processed_pngs", skip_existing: bool = True, cleanup_pngs: bool = True) -> Dict[str, str]:
    """
    Convert Gerber files to PNGs at multiple DPI settings and save as NPZ.
    
    This function processes multiple folders of Gerber files, converting each
    to PNG at various DPI settings, then saving the arrays as compressed NPZ files.
    Optionally cleans up intermediate PNG files after NPZ creation.
    
    Parameters:
        gerber_folders (List[Path]): List of folder paths containing .gbr files
        dpi_list (List[int]): List of DPI values to generate (e.g., [50, 100, 200, 300])
        output_base_dir (str): Base directory for output (default: "Assets/processed_pngs")
        skip_existing (bool): Skip conversion if NPZ file already exists (default: True)
        cleanup_pngs (bool): Delete PNG files after creating NPZ (default: True)
        
    Returns:
        Dict[str, str]: Mapping of "{folder_name}_dpi_{dpi}" -> NPZ file path
        
    Example:
        >>> from pathlib import Path
        >>> folders = [Path("Assets/gerbers/Cu_Balancing_Gerber/Q1")]
        >>> dpi_values = [100, 200, 300]
        >>> results = batch_gerber_to_png_multi_dpi(folders, dpi_values)
        >>> print(results)
        {'Q1_dpi_100': 'Assets/processed_pngs/Q1_dpi_100/Q1_dpi_100.npz', ...}
    """
    # Import required functions (assumes they exist in your codebase)
    # You'll need to add these imports based on where your functions are defined
    from lib.conversion_operations.gerber_converter import gerber_to_png_gerbv
    from lib.conversion_operations.png_loader import bitmap_to_array

    # from your_module import wait_for_file_stability  # Add this import
    
    results = {}
    npz_save_path = "" # Initialize npz_save_path
    save_folder = "" # Initialize Save Folder Path
    for folder, dpi_value in product(gerber_folders, dpi_list):
        # Processing logic
        arrays = {}
        print(f"Processing Gerber files in folder: {folder}")
        
        for file in folder.expanduser().resolve().iterdir():
            if file.suffix.lower() == '.gbr':
                save_folder = f"{output_base_dir}/{folder.name}_dpi_{dpi_value}"
                npz_save_path = f"{save_folder}/{folder.name}_dpi_{dpi_value}.npz"
                
                Path(save_folder).mkdir(parents=True, exist_ok=True)
                png_file_path = Path(save_folder) / f"{file.stem}.png"
                
                if skip_existing and Path(npz_save_path).exists():
                    print(f"NPZ file already exists at {npz_save_path}, skipping conversion.")
                    results[f"{folder.name}_dpi_{dpi_value}"] = npz_save_path
                    continue
                
                if not Path(png_file_path).exists():
                    _, out_file = gerber_to_png_gerbv(
                        gerb_file_path=file,
                        save_folder=save_folder,
                        save_name=file.stem,
                        dpi=dpi_value,
                        anti_alias=True,
                        wait=True
                    )
                else:
                    out_file = png_file_path
                
                # Wait for file stability (uncomment when function is available)
                wait_for_file_stability(out_file, new_name=file.stem)
                
                array = bitmap_to_array(out_file, inverted=False)
                arrays[file.stem] = array
        
        if arrays:
            np.savez_compressed(npz_save_path, **arrays)
            results[f"{folder.name}_dpi_{dpi_value}"] = npz_save_path
            print(f"Saved NPZ file: {npz_save_path}")
        
        # Cleanup PNG files
        if cleanup_pngs:
            for file in glob.glob(os.path.join(save_folder, "*.png")):
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except OSError as e:
                    print(f"Error deleting {file}: {e}")
    
    return results

if __name__ == "__main__":
    folders = [
    Path("Assets/gerbers/Cu_Balancing_Gerber/Q1"),
    Path("Assets/gerbers/Cu_Balancing_Gerber/Q2"),
    Path("Assets/gerbers/Cu_Balancing_Gerber/Q3"),
    Path("Assets/gerbers/Cu_Balancing_Gerber/Q4"),
]
    dpi_values = [25]
    cleanup_pngs = True
    results = batch_gerber_to_png_multi_dpi(folders, dpi_values)
    print(results)

