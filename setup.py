"""
Setup script for Copper Balancing Image Processing package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="copper-balancing",
    version="0.1.0",
    author="Your Name",
    author_email="ace15.guest@gmail.com",
    description="A modular Python framework for processing PCB Gerber files with image processing operations and comparing them to real data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ace15guest/copper-balancing",
    packages=find_packages(include=['src', 'src.*', 'lib', 'lib.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0,<2.0.0",
        "Pillow>=10.0.0,<11.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "copper-balancing=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# Made with Bob
