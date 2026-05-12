"""
Assignment 4 - UCS547: Numba Programming
Run on Google Colab with GPU runtime enabled.
"""

# ============================================================
# Setup - Run this cell first in Colab
# ============================================================
import numpy as np
import time
import math
from numba import jit, cuda, vectorize, float32, float64, int64

# ============================================================
# Q1: Element-wise f(x) = x^2 + 3x + 5 using CUDA kernel
# ============================================================
print("=" * 70)
print("Q1: Element-wise computation f(x) = x^2 + 3x + 5")
print("=" * 70)

N = 5_000_000

# --- CPU version ---
def poly_cpu(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] ** 2 + 3 * x[i] + 5

# --- CUDA kernel (float64) ---
@cuda.jit
def poly_kernel_f64(x, out):
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        out[idx] = x[idx] ** 2 + 3 * x[idx] + 5

# --- CUDA kernel (float32) ---
@cuda.jit
def poly_kernel_f32(x, out):
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        out[idx] = x[idx] ** 2 + 3 * x[idx] + 5

# Generate data
x_cpu = np.random.randn(N).astype(np.float64)
out_cpu = np.zeros(N, dtype=np.float64)

# CPU timing (NumPy vectorized as practical CPU baseline)
start = time.time()
out_numpy = x_cpu ** 2 + 3 * x_cpu + 5
cpu_time = time.time() - start
print(f"CPU (NumPy vectorized) time: {cpu_time:.6f} s")

# --- GPU float64 ---
threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

x_dev64 = cuda.to_device(x_cpu)
out_dev64 = cuda.device_array(N, dtype=np.float64)

# Warm-up
poly_kernel_f64[blocks_per_grid, threads_per_block](x_dev64, out_dev64)
cuda.synchronize()

start = time.time()
poly_kernel_f64[blocks_per_grid, threads_per_block](x_dev64, out_dev64)
cuda.synchronize()
gpu_f64_time = time.time() - start
out_gpu64 = out_dev64.copy_to_host()
print(f"GPU (float64) time:          {gpu_f64_time:.6f} s")
print(f"  Speedup over CPU:          {cpu_time / gpu_f64_time:.2f}x")
print(f"  Results match CPU:         {np.allclose(out_numpy, out_gpu64)}")

# --- Q1a: GPU float32 ---
x_f32 = x_cpu.astype(np.float32)
x_dev32 = cuda.to_device(x_f32)
out_dev32 = cuda.device_array(N, dtype=np.float32)

# Warm-up
poly_kernel_f32[blocks_per_grid, threads_per_block](x_dev32, out_dev32)
cuda.synchronize()

start = time.time()
poly_kernel_f32[blocks_per_grid, threads_per_block](x_dev32, out_dev32)
cuda.synchronize()
gpu_f32_time = time.time() - start
out_gpu32 = out_dev32.copy_to_host()
print(f"GPU (float32) time:          {gpu_f32_time:.6f} s")
print(f"  Speedup over CPU:          {cpu_time / gpu_f32_time:.2f}x")
print(f"  float32 vs float64 GPU:    {gpu_f64_time / gpu_f32_time:.2f}x faster")
print(f"  Results close (f32 vs f64): {np.allclose(out_gpu64, out_gpu32, atol=1e-2)}")

# ============================================================
# Q2: 1-D Histogram Computation
# ============================================================
print("\n" + "=" * 70)
print("Q2: 1-D Histogram Computation (1 million random values)")
print("=" * 70)

NUM_VALS = 1_000_000
NUM_BINS = 50
data = np.random.uniform(0, 1, NUM_VALS)

# --- Pure Python histogram ---
def histogram_python(data, num_bins):
    hist = [0] * num_bins
    bin_width = 1.0 / num_bins
    for val in data:
        b = int(val / bin_width)
        if b >= num_bins:
            b = num_bins - 1
        hist[b] += 1
    return hist

start = time.time()
hist_python = histogram_python(data, NUM_BINS)
python_time = time.time() - start
print(f"Pure Python time:        {python_time:.6f} s")

# --- NumPy histogram ---
start = time.time()
hist_numpy, _ = np.histogram(data, bins=NUM_BINS, range=(0, 1))
numpy_time = time.time() - start
print(f"NumPy time:              {numpy_time:.6f} s")

# --- Numba JIT histogram ---
@jit(nopython=True)
def histogram_numba(data, num_bins):
    hist = np.zeros(num_bins, dtype=np.int64)
    bin_width = 1.0 / num_bins
    for val in data:
        b = int(val / bin_width)
        if b >= num_bins:
            b = num_bins - 1
        hist[b] += 1
    return hist

# Warm-up
_ = histogram_numba(data[:10], NUM_BINS)

start = time.time()
hist_numba = histogram_numba(data, NUM_BINS)
numba_time = time.time() - start
print(f"Numba JIT time:          {numba_time:.6f} s")

# Correctness
print(f"\nPython vs NumPy match:   {np.array_equal(np.array(hist_python), hist_numpy)}")
print(f"Numba vs NumPy match:    {np.array_equal(hist_numba, hist_numpy)}")
print(f"\nSpeedup - Numba over Python: {python_time / numba_time:.2f}x")
print(f"Speedup - NumPy over Python: {python_time / numpy_time:.2f}x")
print(f"Speedup - Numba over NumPy:  {numpy_time / numba_time:.2f}x")

# ============================================================
# Q3: Monte Carlo Pi Estimation
# ============================================================
print("\n" + "=" * 70)
print("Q3: Monte Carlo Pi Estimation")
print("=" * 70)

NSAMPLES = 5_000_000

# --- Q3a: Pure Python version ---
def monte_carlo_pi_python(nsamples):
    inside = 0
    for _ in range(nsamples):
        x = np.random.random()
        y = np.random.random()
        if x ** 2 + y ** 2 <= 1.0:
            inside += 1
    return 4.0 * inside / nsamples

start = time.time()
pi_python = monte_carlo_pi_python(NSAMPLES)
python_pi_time = time.time() - start
print(f"Pure Python: pi = {pi_python:.6f}, time = {python_pi_time:.4f} s")

# --- Q3a: Numba version ---
@jit(nopython=True)
def monte_carlo_pi_numba(nsamples):
    inside = 0
    for i in range(nsamples):
        x = np.random.random()
        y = np.random.random()
        if x ** 2 + y ** 2 <= 1.0:
            inside += 1
    return 4.0 * inside / nsamples

# --- Q3b: First call includes JIT compilation ---
start_first = time.time()
pi_numba_first = monte_carlo_pi_numba(NSAMPLES)
numba_first_time = time.time() - start_first
print(f"Numba (1st call, includes JIT): pi = {pi_numba_first:.6f}, time = {numba_first_time:.4f} s")

# Second call (compiled)
start = time.time()
pi_numba = monte_carlo_pi_numba(NSAMPLES)
numba_pi_time = time.time() - start
print(f"Numba (2nd call, compiled):     pi = {pi_numba:.6f}, time = {numba_pi_time:.4f} s")

print(f"\nSpeedup (Numba compiled vs Python): {python_pi_time / numba_pi_time:.2f}x")
print(f"Actual pi:                          {math.pi:.6f}")

# --- Q3c: Why does first Numba execution take longer? ---
print("\nQ3c Answer: The first Numba execution takes longer because of JIT")
print("(Just-In-Time) compilation overhead. When a @jit-decorated function is")
print("called for the first time, Numba must analyze the Python bytecode, infer")
print("types, and compile it to optimized machine code. This compilation step")
print("adds a one-time cost. Subsequent calls reuse the cached compiled code,")
print("so they run at full native speed without any compilation delay.")

# ============================================================
# Q4: Pixel Brightness Adjustment with @vectorize
# ============================================================
print("\n" + "=" * 70)
print("Q4: Pixel Brightness Adjustment using @vectorize")
print("=" * 70)

NUM_PIXELS = 10_000_000

# --- Q4a: @vectorize (CPU target) ---
@vectorize(['int64(int64)'], target='cpu')
def adjust_brightness_cpu(pixel):
    val = pixel + pixel // 5  # increase by 20%
    if val > 255:
        val = 255
    return val

pixel_data = np.random.randint(0, 256, size=NUM_PIXELS, dtype=np.int64)

# Warm-up
_ = adjust_brightness_cpu(pixel_data[:100])

start = time.time()
result_cpu = adjust_brightness_cpu(pixel_data)
cpu_vec_time = time.time() - start
print(f"@vectorize (cpu) time:      {cpu_vec_time:.6f} s")
print(f"  Sample input:  {pixel_data[:8]}")
print(f"  Sample output: {result_cpu[:8]}")
print(f"  Max value:     {result_cpu.max()} (should be <= 255)")

# --- Q4c: @vectorize with target='parallel' ---
@vectorize(['int64(int64)'], target='parallel')
def adjust_brightness_parallel(pixel):
    val = pixel + pixel // 5  # increase by 20%
    if val > 255:
        val = 255
    return val

# Warm-up
_ = adjust_brightness_parallel(pixel_data[:100])

start = time.time()
result_parallel = adjust_brightness_parallel(pixel_data)
parallel_vec_time = time.time() - start
print(f"\n@vectorize (parallel) time: {parallel_vec_time:.6f} s")
print(f"  Results match CPU:        {np.array_equal(result_cpu, result_parallel)}")
print(f"  Speedup (parallel/cpu):   {cpu_vec_time / parallel_vec_time:.2f}x")

# --- Q4d: What happens if you pass a list instead of NumPy array? ---
print("\nQ4d: Testing with a Python list input...")
try:
    test_list = [100, 200, 250, 50, 0, 255]
    result_list = adjust_brightness_cpu(np.array(test_list, dtype=np.int64))
    print(f"  List converted to array first - result: {result_list}")
except Exception as e:
    print(f"  Error: {e}")

print("\nQ4d Answer: If you pass a Python list instead of a NumPy array, Numba's")
print("@vectorize will attempt to convert it to a NumPy array automatically.")
print("However, this implicit conversion adds overhead and may fail if the list")
print("contains mixed types that cannot be cleanly cast to the declared type.")
print("For best performance and reliability, always pass NumPy arrays with the")
print("correct dtype matching the vectorize signature.")

# ============================================================
# Q5: Binary Logistic Regression with Gradient Descent
# ============================================================
print("\n" + "=" * 70)
print("Q5: Binary Logistic Regression (Gradient Descent)")
print("=" * 70)

# Synthetic data
NUM_SAMPLES = 100_000
NUM_FEATURES = 10
np.random.seed(42)
X = np.random.randn(NUM_SAMPLES, NUM_FEATURES)
true_w = np.random.randn(NUM_FEATURES)
logits = X @ true_w
probs = 1.0 / (1.0 + np.exp(-logits))
y = np.where(probs > 0.5, 1, -1).astype(np.float64)

# Convert labels from {-1, +1} to {0, 1} for logistic regression
y_01 = ((y + 1) / 2).astype(np.float64)

lr = 0.01
n_iters = 200

# --- Q5a: NumPy version ---
def logistic_regression_numpy(X, y, lr, n_iters):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-z))
        error = pred - y
        grad_w = (X.T @ error) / n
        grad_b = np.sum(error) / n
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

start = time.time()
w_np, b_np = logistic_regression_numpy(X, y_01, lr, n_iters)
numpy_lr_time = time.time() - start

preds_np = (1.0 / (1.0 + np.exp(-(X @ w_np + b_np))) > 0.5).astype(np.float64)
acc_np = np.mean(preds_np == y_01) * 100
print(f"NumPy version:  time = {numpy_lr_time:.4f} s, accuracy = {acc_np:.2f}%")

# --- Q5b: Numba JIT version ---
@jit(nopython=True)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

@jit(nopython=True)
def logistic_regression_numba(X, y, lr, n_iters):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        grad_w = (X.T @ error) / n
        grad_b = np.sum(error) / n
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

# Warm-up
_ = logistic_regression_numba(X[:10], y_01[:10], lr, 2)

start = time.time()
w_nb, b_nb = logistic_regression_numba(X, y_01, lr, n_iters)
numba_lr_time = time.time() - start

preds_nb = (1.0 / (1.0 + np.exp(-(X @ w_nb + b_nb))) > 0.5).astype(np.float64)
acc_nb = np.mean(preds_nb == y_01) * 100
print(f"Numba version:  time = {numba_lr_time:.4f} s, accuracy = {acc_nb:.2f}%")

# --- Q5c: Comparison ---
print(f"\nSpeedup (Numba over NumPy): {numpy_lr_time / numba_lr_time:.2f}x")
print(f"Weight difference (L2):     {np.linalg.norm(w_np - w_nb):.8f}")
print(f"Bias difference:            {abs(b_np - b_nb):.8f}")
print(f"Both produce same accuracy: {abs(acc_np - acc_nb) < 0.1}")

# ============================================================
# Q6: CUDA Kernel - Matrix Addition A + B = C (1024 x 1024)
# ============================================================
print("\n" + "=" * 70)
print("Q6: CUDA Matrix Addition (1024 x 1024)")
print("=" * 70)

MAT_SIZE = 1024

@cuda.jit
def matrix_add_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]

# Host data
A = np.random.randn(MAT_SIZE, MAT_SIZE).astype(np.float64)
B = np.random.randn(MAT_SIZE, MAT_SIZE).astype(np.float64)

# CPU reference
start = time.time()
C_cpu = A + B
cpu_mat_time = time.time() - start
print(f"CPU (NumPy) time:    {cpu_mat_time:.6f} s")

# Transfer to device
A_dev = cuda.to_device(A)
B_dev = cuda.to_device(B)
C_dev = cuda.device_array((MAT_SIZE, MAT_SIZE), dtype=np.float64)

# Configure grid
threads_per_block_2d = (16, 16)
blocks_per_grid_2d = (
    (MAT_SIZE + threads_per_block_2d[0] - 1) // threads_per_block_2d[0],
    (MAT_SIZE + threads_per_block_2d[1] - 1) // threads_per_block_2d[1],
)

# Warm-up
matrix_add_kernel[blocks_per_grid_2d, threads_per_block_2d](A_dev, B_dev, C_dev)
cuda.synchronize()

# Timed run
start = time.time()
matrix_add_kernel[blocks_per_grid_2d, threads_per_block_2d](A_dev, B_dev, C_dev)
cuda.synchronize()
gpu_mat_time = time.time() - start

C_gpu = C_dev.copy_to_host()
print(f"GPU (CUDA) time:     {gpu_mat_time:.6f} s")
print(f"Speedup:             {cpu_mat_time / gpu_mat_time:.2f}x")
print(f"Results match:       {np.allclose(C_cpu, C_gpu)}")
print(f"Grid config:         {blocks_per_grid_2d} blocks, {threads_per_block_2d} threads/block")

print("\n" + "=" * 70)
print("All questions completed.")
print("=" * 70)
