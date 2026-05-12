"""
Assignment 3 (UCS547) - CUDA C++, Thrust Parallel Programming & RAPIDS
Run this notebook in Google Colab with GPU runtime enabled.
"""

# ============================================================================
# Setup and Utility Functions
# ============================================================================

import os
import subprocess
import time

def write_file(filename, content):
    """Write content to a file."""
    with open(filename, 'w') as f:
        f.write(content)
    print(f"[+] Written: {filename}")

def compile_cuda(source, output):
    """Compile a CUDA source file with nvcc."""
    result = subprocess.run(
        ['nvcc', '-o', output, source],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Compilation error:\n{result.stderr}")
        return False
    print(f"[+] Compiled: {source} -> {output}")
    return True

def run_binary(binary):
    """Run a compiled binary and print its output."""
    result = subprocess.run(
        [f'./{binary}'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout

# ============================================================================
# Q1: CUDA C/C++ Vector Addition (N=1024)
# ============================================================================

print("=" * 70)
print("Q1: CUDA C/C++ Vector Addition (N=1024)")
print("=" * 70)

q1_code = r"""
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = 0;
            break;
        }
    }
    printf("Q1 - CUDA Vector Addition (N=%d)\n", N);
    printf("Sample results: C[0]=%.1f, C[1]=%.1f, C[1023]=%.1f\n",
           h_C[0], h_C[1], h_C[1023]);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
"""

write_file('q1_vector_add.cu', q1_code)
if compile_cuda('q1_vector_add.cu', 'q1_vector_add'):
    run_binary('q1_vector_add')


# ============================================================================
# Q2: Thrust Vector Addition (N=1024)
# ============================================================================

print("=" * 70)
print("Q2: Vector Addition Using Thrust Library (N=1024)")
print("=" * 70)

q2_code = r"""
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <stdio.h>

#define N 1024

int main() {
    // Create host vectors and initialize
    thrust::host_vector<float> h_A(N), h_B(N), h_C(N);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Transfer to device
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(N);

    // Element-wise addition using thrust::transform with plus functor
    thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(),
                      thrust::plus<float>());

    // Copy result back to host
    h_C = d_C;

    // Verify results
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = 0;
            break;
        }
    }

    printf("Q2 - Thrust Vector Addition (N=%d)\n", N);
    printf("Sample results: C[0]=%.1f, C[1]=%.1f, C[1023]=%.1f\n",
           h_C[0], h_C[1], h_C[1023]);
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");
    return 0;
}
"""

write_file('q2_thrust_add.cu', q2_code)
if compile_cuda('q2_thrust_add.cu', 'q2_thrust_add'):
    run_binary('q2_thrust_add')


# ============================================================================
# Q3: Thrust Dot Product with CPU Comparison (N=1024)
# ============================================================================

print("=" * 70)
print("Q3: Dot Product - Thrust vs CPU (N=1024)")
print("=" * 70)

q3_code = r"""
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <stdio.h>
#include <time.h>

#define N 1024
#define ITERATIONS 1000

int main() {
    // Initialize host vectors
    thrust::host_vector<float> h_A(N), h_B(N);
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ---- CPU Dot Product ----
    clock_t cpu_start = clock();
    float cpu_result = 0.0f;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        cpu_result = 0.0f;
        for (int i = 0; i < N; i++) {
            cpu_result += h_A[i] * h_B[i];
        }
    }
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;

    // ---- Thrust Dot Product ----
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;

    // Warm-up
    float gpu_result = thrust::inner_product(d_A.begin(), d_A.end(),
                                              d_B.begin(), 0.0f);
    cudaDeviceSynchronize();

    clock_t gpu_start = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        gpu_result = thrust::inner_product(d_A.begin(), d_A.end(),
                                            d_B.begin(), 0.0f);
    }
    cudaDeviceSynchronize();
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC * 1000.0;

    printf("Q3 - Dot Product Comparison (N=%d, %d iterations)\n", N, ITERATIONS);
    printf("---------------------------------------------------\n");
    printf("CPU Result:    %.1f\n", cpu_result);
    printf("Thrust Result: %.1f\n", gpu_result);
    printf("CPU Time:      %.4f ms\n", cpu_time);
    printf("Thrust Time:   %.4f ms\n", gpu_time);
    printf("Speedup:       %.2fx\n", cpu_time / gpu_time);
    printf("Verification:  %s\n",
           (cpu_result == gpu_result) ? "PASSED" : "FAILED");
    return 0;
}
"""

write_file('q3_dot_product.cu', q3_code)
if compile_cuda('q3_dot_product.cu', 'q3_dot_product'):
    run_binary('q3_dot_product')


# ============================================================================
# Q4: CUDA Matrix Multiplication (16x16)
# ============================================================================

print("=" * 70)
print("Q4: CUDA Matrix Multiplication (16x16)")
print("=" * 70)

q4_code = r"""
#include <stdio.h>
#include <cuda_runtime.h>

#define DIM 16

/*
 * Why matrix multiplication needs more computation than element-wise addition:
 *
 * - Element-wise addition: C[i][j] = A[i][j] + B[i][j]
 *   Each output element requires exactly 1 addition.
 *   Total operations for NxN matrices: N^2 additions = O(N^2)
 *
 * - Matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j]) for k=0..N-1
 *   Each output element requires N multiplications and N-1 additions.
 *   Total operations for NxN matrices: N^2 * (2N - 1) ~= O(N^3)
 *
 * For 16x16 matrices:
 *   Addition:       256 operations
 *   Multiplication: 256 * 31 = 7,936 operations (~31x more)
 *
 * Matrix multiplication has cubic complexity O(N^3) vs linear O(N^2) for
 * addition because each output element depends on an entire row and column
 * rather than a single pair of corresponding elements.
 */

__global__ void matMul(float *A, float *B, float *C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dim && col < dim) {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++) {
            sum += A[row * dim + k] * B[k * dim + col];
        }
        C[row * dim + col] = sum;
    }
}

int main() {
    int size = DIM * DIM * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize matrices
    for (int i = 0; i < DIM * DIM; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with a 16x16 thread block (one thread per output element)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(1, 1);
    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, DIM);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify: each element of C should be DIM * (1.0 * 2.0) = 32.0
    int correct = 1;
    for (int i = 0; i < DIM * DIM; i++) {
        if (h_C[i] != (float)(DIM * 2)) {
            correct = 0;
            printf("Mismatch at index %d: expected %.1f, got %.1f\n",
                   i, (float)(DIM * 2), h_C[i]);
            break;
        }
    }

    printf("Q4 - CUDA Matrix Multiplication (%dx%d)\n", DIM, DIM);
    printf("C[0][0]=%.1f, C[7][7]=%.1f, C[15][15]=%.1f\n",
           h_C[0], h_C[7*DIM+7], h_C[15*DIM+15]);
    printf("Expected value for each element: %.1f\n", (float)(DIM * 2));
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    printf("\n--- Computation Comparison ---\n");
    printf("Element-wise Addition: %d ops (O(N^2))\n", DIM * DIM);
    printf("Matrix Multiplication: %d ops (O(N^3))\n", DIM * DIM * (2 * DIM - 1));
    printf("Ratio: %.1fx more computation for multiplication\n",
           (float)(DIM * DIM * (2 * DIM - 1)) / (DIM * DIM));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
"""

write_file('q4_matmul.cu', q4_code)
if compile_cuda('q4_matmul.cu', 'q4_matmul'):
    run_binary('q4_matmul')


# ============================================================================
# Q5: Large Vector Addition - CPU vs CUDA vs Thrust vs RAPIDS (N=5,000,000)
# ============================================================================

print("=" * 70)
print("Q5: Vector Addition Comparison (N=5,000,000)")
print("       CPU Sequential vs CUDA vs Thrust vs RAPIDS (CuPy)")
print("=" * 70)

# --- Part A: CPU and CUDA kernel comparison (compiled C/C++) ---

q5_cuda_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 5000000

__global__ void vectorAddKernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C_cpu = (float *)malloc(size);
    h_C_gpu = (float *)malloc(size);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 1000) / 100.0f;
        h_B[i] = (float)(rand() % 1000) / 100.0f;
    }

    // ---- CPU Sequential ----
    struct timespec cpu_start, cpu_end;
    clock_gettime(CLOCK_MONOTONIC, &cpu_start);
    for (int i = 0; i < N; i++) {
        h_C_cpu[i] = h_A[i] + h_B[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &cpu_end);
    double cpu_ms = (cpu_end.tv_sec - cpu_start.tv_sec) * 1000.0 +
                    (cpu_end.tv_nsec - cpu_start.tv_nsec) / 1e6;

    // ---- CUDA Kernel ----
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Warm-up
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEventRecord(start);
    vectorAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cuda_ms = 0;
    cudaEventElapsedTime(&cuda_ms, start, stop);

    cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    int correct = 1;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_C_cpu[i] - h_C_gpu[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }

    printf("CPU_TIME_MS=%.4f\n", cpu_ms);
    printf("CUDA_TIME_MS=%.4f\n", cuda_ms);
    printf("VERIFICATION=%s\n", correct ? "PASSED" : "FAILED");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_cpu); free(h_C_gpu);
    return 0;
}
"""

write_file('q5_cpu_cuda.cu', q5_cuda_code)
compile_cuda('q5_cpu_cuda.cu', 'q5_cpu_cuda')

# --- Part B: Thrust implementation ---

q5_thrust_code = r"""
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 5000000

int main() {
    thrust::host_vector<float> h_A(N), h_B(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 1000) / 100.0f;
        h_B[i] = (float)(rand() % 1000) / 100.0f;
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(N);

    // Warm-up
    thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(),
                      thrust::plus<float>());
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(),
                      thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float thrust_ms = 0;
    cudaEventElapsedTime(&thrust_ms, start, stop);

    printf("THRUST_TIME_MS=%.4f\n", thrust_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
"""

write_file('q5_thrust.cu', q5_thrust_code)
compile_cuda('q5_thrust.cu', 'q5_thrust')

# Run the compiled binaries and parse results
print("\nRunning CPU vs CUDA benchmark...")
output_cuda = ""
try:
    result = subprocess.run(['./q5_cpu_cuda'], capture_output=True, text=True, timeout=60)
    output_cuda = result.stdout
    print(output_cuda)
except Exception as e:
    print(f"Error: {e}")

print("Running Thrust benchmark...")
output_thrust = ""
try:
    result = subprocess.run(['./q5_thrust'], capture_output=True, text=True, timeout=60)
    output_thrust = result.stdout
    print(output_thrust)
except Exception as e:
    print(f"Error: {e}")

# --- Part C: RAPIDS (CuPy) implementation ---
print("\nRunning RAPIDS (CuPy) benchmark...")

import numpy as np
try:
    import cupy as cp

    N_large = 5_000_000
    np.random.seed(42)
    a_np = np.random.rand(N_large).astype(np.float32)
    b_np = np.random.rand(N_large).astype(np.float32)

    # CuPy vector addition
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)

    # Warm-up
    _ = a_cp + b_cp
    cp.cuda.Stream.null.synchronize()

    cupy_start = time.perf_counter()
    c_cp = a_cp + b_cp
    cp.cuda.Stream.null.synchronize()
    cupy_end = time.perf_counter()
    cupy_ms = (cupy_end - cupy_start) * 1000

    print(f"CUPY_TIME_MS={cupy_ms:.4f}")

    # Verify
    c_np = cp.asnumpy(c_cp)
    max_diff = np.max(np.abs(c_np - (a_np + b_np)))
    print(f"CuPy verification max error: {max_diff:.2e}")

except ImportError:
    print("CuPy not installed. Install with: !pip install cupy-cuda12x")
    cupy_ms = None

# --- Parse all timing results and create comparison ---

def parse_time(output, key):
    for line in output.split('\n'):
        if line.startswith(key):
            return float(line.split('=')[1])
    return None

cpu_ms = parse_time(output_cuda, 'CPU_TIME_MS')
cuda_ms = parse_time(output_cuda, 'CUDA_TIME_MS')
thrust_ms = parse_time(output_thrust, 'THRUST_TIME_MS')

print("\n" + "=" * 60)
print("Q5 - Performance Comparison: Vector Addition (N=5,000,000)")
print("=" * 60)
print(f"{'Method':<25} {'Time (ms)':<15} {'Speedup vs CPU':<15}")
print("-" * 55)

times = {}
if cpu_ms is not None:
    times['CPU Sequential'] = cpu_ms
    print(f"{'CPU Sequential':<25} {cpu_ms:<15.4f} {'1.00x':<15}")
if cuda_ms is not None:
    times['CUDA Kernel'] = cuda_ms
    speedup = cpu_ms / cuda_ms if cpu_ms else 0
    print(f"{'CUDA Kernel':<25} {cuda_ms:<15.4f} {speedup:<15.2f}x")
if thrust_ms is not None:
    times['Thrust'] = thrust_ms
    speedup = cpu_ms / thrust_ms if cpu_ms else 0
    print(f"{'Thrust':<25} {thrust_ms:<15.4f} {speedup:<15.2f}x")
if cupy_ms is not None:
    times['RAPIDS (CuPy)'] = cupy_ms
    speedup = cpu_ms / cupy_ms if cpu_ms else 0
    print(f"{'RAPIDS (CuPy)':<25} {cupy_ms:<15.4f} {speedup:<15.2f}x")

# --- Plot comparison graph ---
import matplotlib.pyplot as plt

if times:
    methods = list(times.keys())
    time_values = list(times.values())
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'][:len(methods)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of execution times
    bars = ax1.bar(methods, time_values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Vector Addition (N=5,000,000) - Execution Time')
    ax1.set_yscale('log')
    for bar, val in zip(bars, time_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=15)

    # Speedup chart
    if cpu_ms:
        speedups = [cpu_ms / t for t in time_values]
        bars2 = ax2.bar(methods, speedups, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Speedup Relative to CPU Sequential')
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars2, speedups):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'{val:.1f}x', ha='center', va='bottom', fontsize=9)
        ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig('q5_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[+] Plot saved: q5_performance_comparison.png")


# ============================================================================
# Q6: Thrust Sum of Vector (1..10)
# ============================================================================

print("\n" + "=" * 70)
print("Q6: Thrust - Sum of Vector [1..10]")
print("=" * 70)

q6_code = r"""
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <stdio.h>

int main() {
    const int N = 10;
    thrust::host_vector<int> h_vec(N);

    // Initialize vector with values 1 to 10
    for (int i = 0; i < N; i++) {
        h_vec[i] = i + 1;
    }

    // Transfer to device
    thrust::device_vector<int> d_vec = h_vec;

    // Compute sum using thrust::reduce
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    printf("Q6 - Thrust Sum of Vector\n");
    printf("Vector: [");
    for (int i = 0; i < N; i++) {
        printf("%d%s", h_vec[i], (i < N-1) ? ", " : "");
    }
    printf("]\n");
    printf("Sum = %d\n", sum);
    printf("Expected = %d\n", N * (N + 1) / 2);
    printf("Verification: %s\n", (sum == N * (N + 1) / 2) ? "PASSED" : "FAILED");
    return 0;
}
"""

write_file('q6_thrust_sum.cu', q6_code)
if compile_cuda('q6_thrust_sum.cu', 'q6_thrust_sum'):
    run_binary('q6_thrust_sum')


# ============================================================================
# Q7: Thrust Sort
# ============================================================================

print("=" * 70)
print("Q7: Thrust - Sort Vector Ascending")
print("=" * 70)

q7_code = r"""
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <stdio.h>

int main() {
    const int N = 8;
    int data[] = {7, 2, 9, 1, 5, 3, 8, 4};

    thrust::host_vector<int> h_vec(data, data + N);

    // Print before sorting
    printf("Q7 - Thrust Sort\n");
    printf("Before sorting: [");
    for (int i = 0; i < N; i++) {
        printf("%d%s", h_vec[i], (i < N-1) ? ", " : "");
    }
    printf("]\n");

    // Transfer to device
    thrust::device_vector<int> d_vec = h_vec;

    // Sort in ascending order
    thrust::sort(d_vec.begin(), d_vec.end());

    // Copy back to host
    h_vec = d_vec;

    // Print after sorting
    printf("After sorting:  [");
    for (int i = 0; i < N; i++) {
        printf("%d%s", h_vec[i], (i < N-1) ? ", " : "");
    }
    printf("]\n");

    // Verify sorted order
    int sorted = 1;
    for (int i = 1; i < N; i++) {
        if (h_vec[i] < h_vec[i-1]) {
            sorted = 0;
            break;
        }
    }
    printf("Verification: %s\n", sorted ? "PASSED" : "FAILED");
    return 0;
}
"""

write_file('q7_thrust_sort.cu', q7_code)
if compile_cuda('q7_thrust_sort.cu', 'q7_thrust_sort'):
    run_binary('q7_thrust_sort')


# ============================================================================
# Cleanup temporary files
# ============================================================================
print("\n" + "=" * 70)
print("All 7 questions completed.")
print("=" * 70)

# Optional: cleanup compiled files
# import glob
# for f in glob.glob('q[1-7]*'):
#     if not f.endswith('.py'):
#         os.remove(f)
