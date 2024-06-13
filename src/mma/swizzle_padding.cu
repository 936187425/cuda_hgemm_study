#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>


// This source code from https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/, to explain the swizzle mode vs padding about the solution of bank conflict.
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* func, char const* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10,
                          size_t num_warmups = 10)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

constexpr size_t div_up(size_t a, size_t b) { return (a + b - 1) / b; }


// padding的方法,BLOCK_TILE_SKEW_SIZE_X是控制padding的size.
template <typename T, size_t BLOCK_TILE_SIZE_X = 32,
          size_t BLOCK_TILE_SIZE_Y = 32, size_t BLOCK_TILE_SKEW_SIZE_X = 0>
__global__ void transpose(T* output_matrix, T const* input_matrix, size_t M,
                          size_t N)
{
    // Waste some shared memory to avoid bank conflicts if
    // BLOCK_TILE_SKEW_SIZE_X != 0. 
    __shared__ T
        shm[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X];

    // In some algorithms, such as matrix multiplication,
    // a warp of threads have to access a column of the 2D matrix in the shared
    // memory. Using the conventional index mapping, if the column size is not a
    // multiple of the warp size, there will be bank conflicts. 
    size_t const input_matrix_from_idx_x{threadIdx.x + blockIdx.x * blockDim.x}; //把
    size_t const input_matrix_from_idx_y{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const input_matrix_from_idx{input_matrix_from_idx_x +
                                       input_matrix_from_idx_y * N};
    size_t const shm_to_idx_x{threadIdx.x};
    size_t const shm_to_idx_y{threadIdx.y};

    if ((input_matrix_from_idx_y < M) && (input_matrix_from_idx_x < N))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict.
        shm[shm_to_idx_y][shm_to_idx_x] = input_matrix[input_matrix_from_idx];
    }

    // Make sure the buffer in a block is filled.
    __syncthreads();

    size_t const block_thread_idx{threadIdx.x + threadIdx.y * blockDim.x};
    size_t const shm_from_idx_x{block_thread_idx / BLOCK_TILE_SIZE_Y};
    size_t const shm_from_idx_y{block_thread_idx % BLOCK_TILE_SIZE_Y};
    size_t const output_matrix_to_idx_x{shm_from_idx_y +
                                        blockIdx.y * blockDim.y};
    size_t const output_matrix_to_idx_y{shm_from_idx_x +
                                        blockIdx.x * blockDim.x};
    size_t const output_matrix_to_idx{output_matrix_to_idx_x +
                                      output_matrix_to_idx_y * M};

    if ((output_matrix_to_idx_y < N) && (output_matrix_to_idx_x < M))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict if BLOCK_TILE_SKEW_SIZE_X = 1.
        output_matrix[output_matrix_to_idx] =
            shm[shm_from_idx_y][shm_from_idx_x];
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X = 32,
          size_t BLOCK_TILE_SIZE_Y = 32>
__global__ void transpose_swizzling(T* output_matrix, T const* input_matrix,
                                    size_t M, size_t N)
{
    __shared__ T shm[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_X];

    // In some algorithms, such as matrix multiplication,
    // a warp of threads have to access a column of the 2D matrix in the shared
    // memory. Using the conventional index mapping, if the column size is not a
    // multiple of the warp size, there will be bank conflicts.
    size_t const input_matrix_from_idx_x{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const input_matrix_from_idx_y{threadIdx.y + blockIdx.y * blockDim.y};
    size_t const input_matrix_from_idx{input_matrix_from_idx_x +
                                       input_matrix_from_idx_y * N};
    size_t const shm_to_idx_x{threadIdx.x};
    size_t const shm_to_idx_y{threadIdx.y};
    size_t const shm_to_idx_x_swizzled{(shm_to_idx_x ^ shm_to_idx_y) %
                                       BLOCK_TILE_SIZE_X};

    if ((input_matrix_from_idx_y < M) && (input_matrix_from_idx_x < N))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict.
        shm[shm_to_idx_y][shm_to_idx_x_swizzled] =
            input_matrix[input_matrix_from_idx];
    }

    // Make sure the buffer in a block is filled.
    __syncthreads();

    size_t const block_thread_idx{threadIdx.x + threadIdx.y * blockDim.x};
    size_t const shm_from_idx_x{block_thread_idx / BLOCK_TILE_SIZE_Y};
    size_t const shm_from_idx_y{block_thread_idx % BLOCK_TILE_SIZE_Y};
    size_t const shm_from_idx_x_swizzled{(shm_from_idx_x ^ shm_from_idx_y) %
                                         BLOCK_TILE_SIZE_X};
    size_t const output_matrix_to_idx_x{shm_from_idx_y +
                                        blockIdx.y * blockDim.y};
    size_t const output_matrix_to_idx_y{shm_from_idx_x +
                                        blockIdx.x * blockDim.x};
    size_t const output_matrix_to_idx{output_matrix_to_idx_x +
                                      output_matrix_to_idx_y * M};

    if ((output_matrix_to_idx_y < N) && (output_matrix_to_idx_x < M))
    {
        // Coalesced global memory access.
        // No shared memory bank conflict.
        output_matrix[output_matrix_to_idx] =
            shm[shm_from_idx_y][shm_from_idx_x_swizzled];
    }
}

template <typename T>
void launch_transpose_with_shm_bank_conflict(T* d_output_matrix,
                                             T const* d_input_matrix, size_t M,
                                             size_t N, cudaStream_t stream)
{
    constexpr size_t BLOCK_TILE_SIZE_X{32};
    constexpr size_t BLOCK_TILE_SIZE_Y{32};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{0};
    dim3 const block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y};
    dim3 const grid_size{static_cast<unsigned int>(div_up(N, block_size.x)),
                         static_cast<unsigned int>(div_up(M, block_size.y))};
    transpose<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SKEW_SIZE_X>
        <<<grid_size, block_size, 0, stream>>>(d_output_matrix, d_input_matrix,
                                               M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_transpose_without_shm_bank_conflict_via_padding(
    T* d_output_matrix, T const* d_input_matrix, size_t M, size_t N,  //d_output_matrix [M,N];d_input_matrix [N,M]
    cudaStream_t stream) //example M=8192,N=8192
{
    constexpr size_t BLOCK_TILE_SIZE_X{32};
    constexpr size_t BLOCK_TILE_SIZE_Y{32};
    constexpr size_t BLOCK_TILE_SKEW_SIZE_X{1};
    dim3 const block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y}; //[32,32]
    dim3 const grid_size{static_cast<unsigned int>(div_up(N, block_size.x)), 
                         static_cast<unsigned int>(div_up(M, block_size.y))};
    transpose<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SKEW_SIZE_X>
        <<<grid_size, block_size, 0, stream>>>(d_output_matrix, d_input_matrix,
                                               M, N);
    CHECK_LAST_CUDA_ERROR();
}

// By swizzling method
template <typename T>
void launch_transpose_without_shm_bank_conflict_via_swizzling(
    T* d_output_matrix, T const* d_input_matrix, size_t M, size_t N,
    cudaStream_t stream)
{
    constexpr size_t BLOCK_TILE_SIZE_X{32};
    constexpr size_t BLOCK_TILE_SIZE_Y{32};
    dim3 const block_size{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y};
    dim3 const grid_size{static_cast<unsigned int>(div_up(N, block_size.x)),
                         static_cast<unsigned int>(div_up(M, block_size.y))};
    transpose_swizzling<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y><<<grid_size, block_size, 0, stream>>>(
        d_output_matrix, d_input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
bool is_equal(T const* data_1, T const* data_2, size_t size)
{
    for (size_t i{0}; i < size; ++i)
    {
        if (data_1[i] != data_2[i])
        {
            return false;
        }
    }
    return true;
}

template <typename T>
bool verify_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    // Fixed random seed for reproducibility
    std::mt19937 gen{0};
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    std::vector<T> matrix(matrix_size, 0.0f);
    std::vector<T> matrix_transposed(matrix_size, 1.0f);
    std::vector<T> matrix_transposed_reference(matrix_size, 2.0f);
    std::uniform_real_distribution<T> uniform_dist(-256, 256);
    for (size_t i{0}; i < matrix_size; ++i)
    {
        matrix[i] = uniform_dist(gen);
    }
    // Create the reference transposed matrix using CPU.
    for (size_t i{0}; i < M; ++i)
    {
        for (size_t j{0}; j < N; ++j)
        {
            size_t const from_idx{i * N + j};
            size_t const to_idx{j * M + i};
            matrix_transposed_reference[to_idx] = matrix[from_idx];
        }
    }
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix.data(),
                                matrix_size * sizeof(T),
                                cudaMemcpyHostToDevice));
    transpose_function(d_matrix_transposed, d_matrix, M, N, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaMemcpy(matrix_transposed.data(), d_matrix_transposed,
                                matrix_size * sizeof(T),
                                cudaMemcpyDeviceToHost));
    bool const correctness{is_equal(matrix_transposed.data(),
                                    matrix_transposed_reference.data(),
                                    matrix_size)};
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    return correctness;
}

template <typename T>
float profile_transpose_implementation(
    std::function<void(T*, T const*, size_t, size_t, cudaStream_t)>
        transpose_function,
    size_t M, size_t N)
{
    constexpr int num_repeats{100};
    constexpr int num_warmups{10};
    cudaStream_t stream;
    size_t const matrix_size{M * N};
    T* d_matrix;
    T* d_matrix_transposed;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_transposed, matrix_size * sizeof(T)));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::function<void(cudaStream_t)> const transpose_function_wrapped{
        std::bind(transpose_function, d_matrix_transposed, d_matrix, M, N,
                  std::placeholders::_1)};
    float const transpose_function_latency{measure_performance(
        transpose_function_wrapped, stream, num_repeats, num_warmups)};
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_transposed));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    return transpose_function_latency;
}

void print_latencty(std::string const& kernel_name, float latency)
{
    std::cout << kernel_name << ": " << std::fixed << std::setprecision(2)
              << latency << " ms" << std::endl;
}

int main()
{
    // Unit tests.
    for (size_t m{1}; m <= 64; ++m) //size_t是一个无符号的整数，有足够的数值范围来表示任何对象的大小。
    {
        for (size_t n{1}; n <= 64; ++n)
        {
            assert(verify_transpose_implementation<float>(
                &launch_transpose_with_shm_bank_conflict<float>, m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_without_shm_bank_conflict_via_padding<float>,
                m, n));
            assert(verify_transpose_implementation<float>(
                &launch_transpose_without_shm_bank_conflict_via_swizzling<
                    float>,
                m, n));
        }
    }

    // M: Number of rows.
    size_t const M{8192};
    // N: Number of columns.
    size_t const N{8192};
    std::cout << M << " x " << N << " Matrix" << std::endl;
    float const latency_with_shm_bank_conflict{
        profile_transpose_implementation<float>(
            &launch_transpose_with_shm_bank_conflict<float>, M, N)};
    print_latencty("Transpose with Shared Memory Bank Conflict",
                   latency_with_shm_bank_conflict);
    float const latency_without_shm_bank_conflict_via_padding{
        profile_transpose_implementation<float>(
            &launch_transpose_without_shm_bank_conflict_via_padding<float>, M,
            N)};
    print_latencty("Transpose without Shared Memory Bank Conflict via Padding",
                   latency_without_shm_bank_conflict_via_padding);
    float const latency_without_shm_bank_conflict_via_swizzling{
        profile_transpose_implementation<float>(
            &launch_transpose_without_shm_bank_conflict_via_swizzling<float>, M,
            N)};
    print_latencty(
        "Transpose without Shared Memory Bank Conflict via Swizzling",
        latency_without_shm_bank_conflict_via_swizzling);

    return 0;
}
