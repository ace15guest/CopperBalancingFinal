"""
Gerber to PNG Converter

This module handles conversion of Gerber files to PNG images using gerbv.
"""

import subprocess
from pathlib import Path
from typing import List
from src.config import Config


def gerber_to_png_gerbv(gerb_file_path, save_folder, save_name, dpi=300, outline_path=None, anti_alias=False, wait=False):
    """
    Converts a Gerber file to PNG format using gerbv command line tool.

    Parameters:
    gerb_file_path (str): Path to the input Gerber file.
    save_folder (str): Folder where the output PNG will be saved.
    save_name (str): Name of the output PNG file (without extension).
    dpi (int): Resolution of the output PNG file. Default is 300.

    Returns:
    None
    """

    gerb_file_path = Path(gerb_file_path).expanduser().resolve() # The gerber file path
    outline_path = Path(outline_path).expanduser().resolve() if outline_path else None # The outline file path

    # Create the save folder if it doesn't exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    # What the output png file will be named and where it will be saved
    out_name = f"{save_name}.png"
    out_png_path = Path(save_folder) / out_name

    gerbv_path = Path("Assets/gerbv/gerbv.exe")  # Path to gerbv executable

    if gerbv_path.exists():
        cmd = [str(gerbv_path), "-x", "png", "-D", str(dpi)]
        if anti_alias:
            cmd.extend(["-a"])
        cmd.extend(["-o", str(out_png_path), str(gerb_file_path)])
        if outline_path:
            cmd.append(str(outline_path))
    else:
        raise FileNotFoundError(f"gerbv executable not found at {gerbv_path}")
    
    if wait:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ' '.join(cmd), out_png_path


def batch_gerber_to_png(gerber_dir: str, output_dir: str, dpi: int = 300) -> List[str]:
    """
    Convert all Gerber files in a directory to PNG.
    
    Parameters:
        gerber_dir (str): Directory containing Gerber files
        output_dir (str): Directory for output PNG files
        dpi (int, optional): Resolution. Default: 300
        
    Returns:
        list: List of generated PNG file paths
    """
    # TODO: Implement batch Gerber to PNG conversion
    raise NotImplementedError("Batch Gerber conversion not yet implemented")


def get_gerbv_path() -> str:
    """
    Get the path to gerbv.exe from configuration.
    
    Returns:
        str: Absolute path to gerbv.exe
    """
    return str(Config.GERBV_PATH)

# Made with Bob
