"""
Tests for blur_filters module

Tests blur functions (box_blur, blur_2d_gaussian, blur_call) on data from
Q1, Q2, Q3, Q4, and Global quadrant folders.
"""

import pytest
import numpy as np
import os
from pathlib import Path

from lib.array_operations.blur_filters import blur_call, box_blur, blur_2d_gaussian
from lib.conversion_operations.npz_handler import load_npz


# Test data paths - using processed PNG NPZ files
ASSETS_DIR = Path("Assets/processed_pngs")
QUADRANTS = ["Q1", "Q2", "Q3", "Q4", "Global"]
DPI_SETTINGS = {
    "Q1": [50, 100, 200, 250, 300, 500, 700],
    "Q2": [50, 100, 200, 250, 300, 500, 700],
    "Q3": [50, 100, 200, 250, 300, 500, 700],
    "Q4": [100, 200, 250],  # Q4 has fewer DPI options
    "Global": [100, 200, 250, 300]
}


def get_test_data_paths():
    """Get all available NPZ file paths for testing."""
    paths = []
    for quadrant in QUADRANTS:
        for dpi in DPI_SETTINGS.get(quadrant, []):
            npz_path = ASSETS_DIR / f"{quadrant}_dpi_{dpi}" / f"{quadrant}_dpi_{dpi}.npz"
            if npz_path.exists():
                paths.append((quadrant, dpi, npz_path))
    return paths


def load_sample_array(npz_path, max_layers=3):
    """Load a sample array from NPZ file for testing."""
    try:
        data = load_npz(str(npz_path))
        # Get first available array from the NPZ file
        for key, array in data.items():
            if isinstance(array, np.ndarray) and array.ndim == 2:
                return array
        return None
    except Exception as e:
        pytest.skip(f"Could not load data from {npz_path}: {e}")


class TestBlurCall:
    """Test the blur_call function with different quadrants."""
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths())
    def test_blur_call_box_blur(self, quadrant, dpi, npz_path):
        """Test blur_call with box_blur on all quadrants."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = blur_call(array, blur_type="box_blur", radius=5)
        
        assert result.shape == array.shape, f"Shape mismatch for {quadrant} at {dpi} DPI"
        assert result.dtype == array.dtype, f"Dtype mismatch for {quadrant} at {dpi} DPI"
        assert not np.array_equal(result, array), f"Blur had no effect on {quadrant} at {dpi} DPI"
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths())
    def test_blur_call_gaussian(self, quadrant, dpi, npz_path):
        """Test blur_call with Gaussian blur on all quadrants."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = blur_call(array, blur_type="gauss", sigma=2.0)
        
        assert result.shape == array.shape, f"Shape mismatch for {quadrant} at {dpi} DPI"
        assert result.dtype == array.dtype, f"Dtype mismatch for {quadrant} at {dpi} DPI"
        assert not np.array_equal(result, array), f"Blur had no effect on {quadrant} at {dpi} DPI"
    
    def test_blur_call_invalid_type(self):
        """Test blur_call with invalid blur type."""
        array = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="Unknown blur_type"):
            blur_call(array, blur_type="invalid_blur")
    
    def test_blur_call_invalid_dimensions(self):
        """Test blur_call with non-2D array."""
        array = np.random.rand(10, 10, 10)  # 3D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            blur_call(array, blur_type="box_blur")


class TestBoxBlur:
    """Test the box_blur function with different quadrants."""
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths())
    def test_box_blur_default_params(self, quadrant, dpi, npz_path):
        """Test box_blur with default parameters on all quadrants."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = box_blur(array)
        
        assert result.shape == array.shape, f"Shape mismatch for {quadrant} at {dpi} DPI"
        assert result.dtype == array.dtype, f"Dtype mismatch for {quadrant} at {dpi} DPI"
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths()[:5])  # Test subset
    @pytest.mark.parametrize("window_size", [3, 5, 7, 11])
    def test_box_blur_various_window_sizes(self, quadrant, dpi, npz_path, window_size):
        """Test box_blur with various window sizes."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = box_blur(array, window_size=window_size)
        
        assert result.shape == array.shape
        assert result.dtype == array.dtype
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths()[:3])  # Test subset
    @pytest.mark.parametrize("mode", ["nearest", "reflect", "mirror", "wrap"])
    def test_box_blur_boundary_modes(self, quadrant, dpi, npz_path, mode):
        """Test box_blur with different boundary modes."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = box_blur(array, window_size=5, mode=mode)
        
        assert result.shape == array.shape
        assert result.dtype == array.dtype
    
    def test_box_blur_invalid_window_size(self):
        """Test box_blur with invalid window size."""
        array = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            box_blur(array, window_size=0)
    
    def test_box_blur_preserves_integer_dtype(self):
        """Test that box_blur preserves integer dtypes."""
        array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = box_blur(array, window_size=5)
        assert result.dtype == np.uint8


class TestGaussianBlur:
    """Test the blur_2d_gaussian function with different quadrants."""
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths())
    def test_gaussian_blur_default_params(self, quadrant, dpi, npz_path):
        """Test Gaussian blur with default parameters on all quadrants."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = blur_2d_gaussian(array)
        
        assert result.shape == array.shape, f"Shape mismatch for {quadrant} at {dpi} DPI"
        assert result.dtype == array.dtype, f"Dtype mismatch for {quadrant} at {dpi} DPI"
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths()[:5])  # Test subset
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_gaussian_blur_various_sigmas(self, quadrant, dpi, npz_path, sigma):
        """Test Gaussian blur with various sigma values."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = blur_2d_gaussian(array, sigma=sigma)
        
        assert result.shape == array.shape
        assert result.dtype == array.dtype
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths()[:3])  # Test subset
    def test_gaussian_blur_anisotropic(self, quadrant, dpi, npz_path):
        """Test Gaussian blur with different sigma per axis."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = blur_2d_gaussian(array, sigma=(1.0, 3.0))
        
        assert result.shape == array.shape
        assert result.dtype == array.dtype
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths()[:3])  # Test subset
    @pytest.mark.parametrize("mode", ["nearest", "reflect", "mirror", "wrap"])
    def test_gaussian_blur_boundary_modes(self, quadrant, dpi, npz_path, mode):
        """Test Gaussian blur with different boundary modes."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        result = blur_2d_gaussian(array, sigma=2.0, mode=mode)
        
        assert result.shape == array.shape
        assert result.dtype == array.dtype
    
    def test_gaussian_blur_invalid_sigma(self):
        """Test Gaussian blur with invalid sigma."""
        array = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="sigma must be positive"):
            blur_2d_gaussian(array, sigma=0)
        with pytest.raises(ValueError, match="sigma must be positive"):
            blur_2d_gaussian(array, sigma=-1.0)
    
    def test_gaussian_blur_invalid_sigma_tuple(self):
        """Test Gaussian blur with invalid sigma tuple."""
        array = np.random.rand(100, 100)
        with pytest.raises(ValueError, match="sigma tuple must have 2 positive values"):
            blur_2d_gaussian(array, sigma=(1.0, 0))
    
    def test_gaussian_blur_preserves_integer_dtype(self):
        """Test that Gaussian blur preserves integer dtypes."""
        array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = blur_2d_gaussian(array, sigma=2.0)
        assert result.dtype == np.uint8
    
    def test_gaussian_blur_no_preserve_dtype(self):
        """Test Gaussian blur without dtype preservation."""
        array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = blur_2d_gaussian(array, sigma=2.0, preserve_dtype=False)
        assert result.dtype == np.float64


class TestBlurComparison:
    """Compare blur results across quadrants."""
    
    def test_blur_consistency_across_quadrants(self):
        """Test that blur produces consistent results for similar data."""
        # Create synthetic test data
        test_array = np.random.rand(100, 100)
        
        # Apply same blur to the array multiple times
        results = []
        for _ in range(5):
            result = blur_call(test_array.copy(), blur_type="box_blur", radius=5)
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(results[0], results[i])
    
    @pytest.mark.parametrize("quadrant,dpi,npz_path", get_test_data_paths()[:3])
    def test_box_vs_gaussian_blur_difference(self, quadrant, dpi, npz_path):
        """Test that box blur and Gaussian blur produce different results."""
        array = load_sample_array(npz_path)
        if array is None:
            pytest.skip(f"No valid array found in {npz_path}")
        
        box_result = blur_call(array, blur_type="box_blur", radius=5)
        gauss_result = blur_call(array, blur_type="gauss", sigma=2.0)
        
        # Results should be different
        assert not np.array_equal(box_result, gauss_result)


class TestBlurEdgeCases:
    """Test edge cases for blur functions."""
    
    def test_blur_on_uniform_array(self):
        """Test blur on uniform array (all same values)."""
        array = np.ones((100, 100)) * 128
        
        box_result = box_blur(array, window_size=5)
        gauss_result = blur_2d_gaussian(array, sigma=2.0)
        
        # Uniform array should remain uniform after blur
        np.testing.assert_array_almost_equal(box_result, array)
        np.testing.assert_array_almost_equal(gauss_result, array)
    
    def test_blur_on_small_array(self):
        """Test blur on very small arrays."""
        array = np.random.rand(5, 5)
        
        box_result = box_blur(array, window_size=3)
        gauss_result = blur_2d_gaussian(array, sigma=1.0)
        
        assert box_result.shape == array.shape
        assert gauss_result.shape == array.shape
    
    def test_blur_with_window_larger_than_array(self):
        """Test blur with window size larger than array."""
        array = np.random.rand(10, 10)
        
        # Should still work, just heavily blurred
        result = box_blur(array, window_size=15)
        assert result.shape == array.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
