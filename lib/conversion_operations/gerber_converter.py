"""
Gerber to PNG Converter

This module handles conversion of Gerber files to PNG images using gerbv.
"""
import os
import subprocess
import time
from pathlib import Path
from typing import List
from src.config import Config

def wait_for_file_stability(file_path, check_interval=.1, stability_checks=50, new_name=None, trial_times=300):
    """
    Waits until a file exists and its size remains stable for a given number of checks.

    Args:
        file_path (str): The path to the file to monitor.
        check_interval (int): The time in seconds to wait between checks.
        stability_checks (int): The number of consecutive checks for stable file size.
    """
    print(f"Waiting for file: {file_path}")

    # Wait for the file to exist
    count=0
    while not os.path.exists(file_path):
        time.sleep(check_interval)
        count+=1
        print(f"File not found, waiting... {file_path} {count}/{trial_times}")
        if count> trial_times:
            print("File not downloading")
            return
    print(f"File found: {file_path}")

    # Wait for the file size to stabilize
    previous_size = -1
    stable_count = 0
    while stable_count < stability_checks:
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            stable_count += 1
            print(f"File size stable ({current_size} bytes). Stability check {stable_count}/{stability_checks}")
        else:
            stable_count = 0  # Reset if size changed
            print(f"File size changed from {previous_size} to {current_size} bytes. Resetting stability checks.")
        previous_size = current_size
        time.sleep(check_interval)
    # os.rename(file_path, f"{'/'.join([file_path.split('/')[0], new_name])}.pdf")
    print(f"File {file_path} is stable.")
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
