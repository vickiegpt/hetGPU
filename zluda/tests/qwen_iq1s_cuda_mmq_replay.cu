#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t status_ = (expr);                                           \
        if (status_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s\n", #expr,                    \
                         cudaGetErrorString(status_));                          \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    const auto size = input.tellg();
    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!input) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(2);
    }
    return bytes;
}

static std::vector<int8_t> read_grid(const std::string &path) {
    std::ifstream input(path);
    std::vector<int8_t> grid;
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() != 16) {
            std::fprintf(stderr, "invalid IQ1_S grid line\n");
            std::exit(2);
        }
        const uint64_t word = std::strtoull(line.c_str(), nullptr, 16);
        for (int byte = 0; byte < 8; ++byte) {
            grid.push_back(static_cast<int8_t>(word >> (8 * byte)));
        }
    }
    if (grid.size() != 2048 * 8) {
        std::fprintf(stderr, "IQ1_S grid must contain 2048 entries\n");
        std::exit(2);
    }
    return grid;
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t *bytes) {
    return static_cast<uint16_t>(bytes[0]) |
           static_cast<uint16_t>(bytes[1]) << 8;
}

__global__ void qwen_iq1s_cuda_mmq(const uint8_t *matrix,
                                   const uint8_t *activation,
                                   const int8_t *grid, float *output) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 1024) {
        return;
    }
    const uint8_t *row_bytes = matrix + row * 800;
    float sum = 0.0f;
    for (int block = 0; block < 16; ++block) {
        const uint8_t *weight = row_bytes + block * 50;
        const float d = __half2float(__ushort_as_half(load_u16(weight)));
        for (int group = 0; group < 8; ++group) {
            const uint16_t qh = load_u16(weight + 34 + 2 * group);
            const int odd_scale = 2 * ((qh >> 12) & 7) + 1;
            const int sign = (qh & 0x8000) ? -1 : 1;
            const int q8_index = block * 8 + group;
            const uint8_t *record = activation + (q8_index / 4) * 144;
            const int subblock = q8_index % 4;
            const __half q8_d = __ushort_as_half(load_u16(record + subblock * 4));
            const __half q8_s =
                __ushort_as_half(load_u16(record + subblock * 4 + 2));
            const int8_t *q8 = reinterpret_cast<const int8_t *>(
                record + 16 + subblock * 32);
            int encoded_grid_dot = 0;
            for (int position = 0; position < 4; ++position) {
                const int index = weight[2 + 4 * group + position] |
                                  (((qh >> (3 * position)) & 7) << 8);
                for (int element = 0; element < 8; ++element) {
                    encoded_grid_dot +=
                        (static_cast<int>(grid[index * 8 + element]) + 1) *
                        static_cast<int>(q8[position * 8 + element]);
                }
            }
            const float d1q = d * odd_scale;
            const float delta = -1.0f + sign * 0.125f;
            const __half2 weight_ds = __floats2half2_rn(d1q, d1q * delta);
            const __half2 activation_ds = __halves2half2(q8_d, q8_s);
            const float2 products = __half22float2(__hmul2(weight_ds, activation_ds));
            sum += encoded_grid_dot * products.x + products.y;
        }
    }
    output[row] = sum;
}

static const float *as_f32(const std::vector<uint8_t> &bytes) {
    return reinterpret_cast<const float *>(bytes.data());
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s CAPTURE_DIR GRID_MEMH OUTPUT_BIN\n", argv[0]);
        return 2;
    }
    const std::string root = argv[1];
    const auto matrix = read_file(root + "/matrix.iq1s.bin");
    const auto activation = read_file(root + "/activation.q8_1.bin");
    const auto u250 = read_file(root + "/actual.f32.bin");
    const auto cpu = read_file(root + "/reference.f32.bin");
    const auto grid = read_grid(argv[2]);
    if (matrix.size() != 819200 || activation.size() != 4608 ||
        u250.size() != 4096 || cpu.size() != 4096) {
        std::fprintf(stderr, "capture extent mismatch\n");
        return 2;
    }

    uint8_t *d_matrix = nullptr;
    uint8_t *d_activation = nullptr;
    int8_t *d_grid = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_matrix, matrix.size()));
    CUDA_CHECK(cudaMalloc(&d_activation, activation.size()));
    CUDA_CHECK(cudaMalloc(&d_grid, grid.size()));
    CUDA_CHECK(cudaMalloc(&d_output, 4096));
    CUDA_CHECK(cudaMemcpy(d_matrix, matrix.data(), matrix.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_activation, activation.data(), activation.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid, grid.data(), grid.size(), cudaMemcpyHostToDevice));
    qwen_iq1s_cuda_mmq<<<4, 256>>>(d_matrix, d_activation, d_grid, d_output);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> output(1024);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, 4096, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::ofstream binary(argv[3], std::ios::binary | std::ios::trunc);
    binary.write(reinterpret_cast<const char *>(output.data()), 4096);
    binary.close();
    size_t u250_pass = 0;
    size_t cpu_pass = 0;
    float max_u250_error = 0.0f;
    float max_cpu_error = 0.0f;
    for (size_t index = 0; index < output.size(); ++index) {
        const float u250_error = std::fabs(output[index] - as_f32(u250)[index]);
        const float cpu_error = std::fabs(output[index] - as_f32(cpu)[index]);
        max_u250_error = std::fmax(max_u250_error, u250_error);
        max_cpu_error = std::fmax(max_cpu_error, cpu_error);
        u250_pass += u250_error <= 1.0e-4f + 1.0e-3f * std::fabs(output[index]);
        cpu_pass += cpu_error <= 1.0e-4f + 1.0e-3f * std::fabs(output[index]);
    }
    std::printf(
        "rows=1024 u250_within_tolerance=%zu cpu_dequant_within_tolerance=%zu "
        "max_u250_error=%.9g max_cpu_error=%.9g "
        "sample42_cuda=%.9g sample42_u250=%.9g sample42_cpu=%.9g\n",
        u250_pass, cpu_pass, max_u250_error, max_cpu_error, output[42],
        as_f32(u250)[42], as_f32(cpu)[42]);

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_grid));
    CUDA_CHECK(cudaFree(d_activation));
    CUDA_CHECK(cudaFree(d_matrix));
    return u250_pass == output.size() ? 0 : 1;
}
