"""
Test Script for Batch Gerber to PNG Conversion

This script demonstrates how to use the batch_gerber_to_png_multi_dpi function
to convert Gerber files to PNGs at multiple DPI settings.

Run this script from the project root directory:
    python examples/test_batch_conversion.py
"""

from pathlib import Path
from lib.gerber_operations.batch_processor import batch_gerber_to_png_multi_dpi
from src.config import Config



def main():
    """Main test function."""
    print("=" * 80)
    print("Batch Gerber to PNG Conversion Test")
    print("=" * 80)
    print()
    
    # Define the base directory for Gerber files
    gerber_base = Config.ASSETS_DIR / "gerbers/Cu_Balancing_Gerber"

    
    # Define folders to process (Q1, Q2, Q3, Q4)
    gerber_folders = [
        gerber_base / "Q1",
        gerber_base / "Q2",
        gerber_base / "Q3",
        gerber_base / "Q4",
    ]
    
    # Check which folders exist
    existing_folders = [f for f in gerber_folders if f.exists()]
    
    if not existing_folders:
        print("ERROR: No Gerber folders found!")
        print(f"Expected folders in: {gerber_base}")
        print("Please check that your Gerber files are in the correct location.")
        return
    
    print(f"Found {len(existing_folders)} folders to process:")
    for folder in existing_folders:
        gbr_count = len(list(folder.glob("*.gbr")))
        print(f"  - {folder.name}: {gbr_count} .gbr files")
    print()
    
    # Define DPI values to generate
    # Start with a small subset for testing, then expand
    dpi_values = [25]  # Start small for testing
    # Full list: [50, 100, 200, 250, 300, 500, 700]
    
    print(f"DPI values to generate: {dpi_values}")
    print()
    
    # Ask for confirmation
    response = input("Proceed with conversion? (y/n): ").strip().lower()
    if response != 'y':
        print("Conversion cancelled.")
        return
    
    print()
    print("Starting batch conversion...")
    print("-" * 80)
    
    try:
        # Run the batch conversion
        results = batch_gerber_to_png_multi_dpi(
            gerber_folders=existing_folders,
            dpi_list=dpi_values,
            output_base_dir="Assets/processed_pngs",
            skip_existing=True,  # Skip if NPZ already exists
            cleanup_pngs=True    # Delete PNGs after creating NPZ
        )
        
        print()
        print("-" * 80)
        print("Conversion complete!")
        print()
        print(f"Generated {len(results)} NPZ files:")
        for key, npz_path in results.items():
            print(f"  ✓ {key}: {npz_path}")
        
    except Exception as e:
        print()
        print("ERROR during conversion:")
        print(f"  {type(e).__name__}: {e}")
        print()
        print("This might be due to missing dependencies or functions.")
        print("Make sure all required functions are properly imported in batch_processor.py")
        return
    
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Made with Bob
