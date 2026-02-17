#!/usr/bin/env python
"""
Launch the Parameter Sweep Analysis Dashboard

This script starts the interactive web dashboard for visualizing
parameter sweep results from Gerber processing experiments.

Usage:
    python run_dashboard.py
    
Then open your browser to: http://127.0.0.1:8050
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lib.visualization import run_dashboard


def main():
    """Main entry point for dashboard."""
    print("=" * 60)
    print("  Parameter Sweep Analysis Dashboard")
    print("=" * 60)
    print()
    
    # Configuration
    data_path = "Assets/DataOutput/data_out.csv"
    host = "127.0.0.1"
    port = 8050
    debug = True
    
    # Check if data file exists
    if not Path(data_path).exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print()
        print("Please ensure your parameter sweep results are saved to:")
        print(f"  {Path(data_path).absolute()}")
        print()
        sys.exit(1)
    
    # Run dashboard
    try:
        run_dashboard(
            data_path=data_path,
            host=host,
            port=port,
            debug=debug
        )
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting dashboard: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Made with Bob
