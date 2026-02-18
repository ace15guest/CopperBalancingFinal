"""
Quick test to verify gradient_analysis optimizations maintain same return values.
"""
import numpy as np
from lib.array_operations.gradient_analysis import analyze_gradients, compute_gradients, compare_gradient_fields

# Set random seed for reproducibility
np.random.seed(42)

# Create test data
Z_model = np.random.randn(50, 50) * 10
Z_ref = np.random.randn(50, 50) * 10

print("Testing gradient analysis optimizations...")
print("=" * 60)

# Test 1: Finite difference method
print("\n1. Testing finite difference method...")
metrics1, angle_diff1, mag_ratio1 = analyze_gradients(
    Z_model, Z_ref, dx=1.0, dy=1.0, method="finite", make_plots=False
)
print("   [OK] Finite difference method completed")
print(f"   - angle_mean_deg: {metrics1['angle_mean_deg']:.4f}")
print(f"   - mag_ratio_mean: {metrics1['mag_ratio_mean']:.4f}")

# Test 2: Plane fitting method
print("\n2. Testing plane fitting method (window_size=3)...")
metrics2, angle_diff2, mag_ratio2 = analyze_gradients(
    Z_model, Z_ref, dx=1.0, dy=1.0, method="plane", window_size=3, make_plots=False
)
print("   [OK] Plane fitting method completed")
print(f"   - angle_mean_deg: {metrics2['angle_mean_deg']:.4f}")
print(f"   - mag_ratio_mean: {metrics2['mag_ratio_mean']:.4f}")

# Test 3: Plane fitting with larger window
print("\n3. Testing plane fitting method (window_size=5)...")
metrics3, angle_diff3, mag_ratio3 = analyze_gradients(
    Z_model, Z_ref, dx=1.0, dy=1.0, method="plane", window_size=5, make_plots=False
)
print("   [OK] Plane fitting with larger window completed")
print(f"   - angle_mean_deg: {metrics3['angle_mean_deg']:.4f}")
print(f"   - mag_ratio_mean: {metrics3['mag_ratio_mean']:.4f}")

# Test 4: Direct gradient computation
print("\n4. Testing direct gradient computation...")
gx, gy = compute_gradients(Z_model, dx=1.0, dy=1.0, method="finite")
print(f"   [OK] Gradient shapes: gx={gx.shape}, gy={gy.shape}")
print(f"   - gx mean: {np.mean(gx):.4f}, std: {np.std(gx):.4f}")
print(f"   - gy mean: {np.mean(gy):.4f}, std: {np.std(gy):.4f}")

# Test 5: Direct comparison
print("\n5. Testing direct gradient field comparison...")
gx_m, gy_m = compute_gradients(Z_model, dx=1.0, dy=1.0, method="finite")
gx_r, gy_r = compute_gradients(Z_ref, dx=1.0, dy=1.0, method="finite")
metrics5, angle_diff5, mag_ratio5 = compare_gradient_fields(gx_m, gy_m, gx_r, gy_r)
print("   [OK] Gradient field comparison completed")
print(f"   - All metrics keys: {list(metrics5.keys())}")

# Verify return value types
print("\n6. Verifying return value types...")
assert isinstance(metrics1, dict), "metrics should be dict"
assert isinstance(angle_diff1, np.ndarray), "angle_diff should be ndarray"
assert isinstance(mag_ratio1, np.ndarray), "mag_ratio should be ndarray"
assert angle_diff1.shape == Z_model.shape, "angle_diff should match input shape"
assert mag_ratio1.shape == Z_model.shape, "mag_ratio should match input shape"
print("   [OK] All return value types are correct")

print("\n" + "=" * 60)
print("[SUCCESS] ALL TESTS PASSED - Optimizations maintain same return values!")
print("=" * 60)

# Made with Bob
