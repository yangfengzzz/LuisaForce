//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "utils/data_type_util.h"
#include "utils/status_util.h"
#include "runtime/buffer.h"
#include "runtime/stream.h"
#include "runtime/ext/metal/metal_command.h"
#include <spdlog/fmt/fmt.h>

#include "benchmark_api.h"

struct ShaderCode {
    const char *name;// Shader case name
    int M0;
    int N0;
    int K0;
    int wg_size_x;
    int wg_size_y;
    DataType input_type; // LHS & RHS element type
    DataType output_type;// Output/Result matrix element type
};

#define SHADER_I8(M0, N0, K0, X, Y)           \
    ShaderCode {                              \
        "Tile[" #M0 "x" #N0 "x" #K0 "]",      \
            M0, N0, K0,                       \
            X, Y, DataType::i8, DataType::i32 \
    }

#define WORKGROUP_TILE_N_I8(X, Y, N0)                           \
    SHADER_I8(4, N0, 4, X, Y), SHADER_I8(8, N0, 4, X, Y),       \
        SHADER_I8(16, N0, 4, X, Y), SHADER_I8(32, N0, 4, X, Y), \
        SHADER_I8(4, N0, 8, X, Y), SHADER_I8(8, N0, 8, X, Y),   \
        SHADER_I8(16, N0, 8, X, Y), SHADER_I8(32, N0, 8, X, Y), \
        SHADER_I8(4, N0, 16, X, Y), SHADER_I8(8, N0, 16, X, Y), \
        SHADER_I8(16, N0, 16, X, Y), SHADER_I8(32, N0, 16, X, Y)

static ShaderCode kShaderCodeCases[] = {
    WORKGROUP_TILE_N_I8(16, 1, 32),
    WORKGROUP_TILE_N_I8(16, 1, 64),
};

/// Fills the 2D matrix with values produced by the |generator| function.
template<typename GeneratorFn>
static void fill_buffer(DataType data_type, void *raw_buffer,
                        unsigned dim_1, unsigned dim_2, GeneratorFn generator) {
    auto fill = [&](auto traits) {
        using Traits = decltype(traits);
        using StorageType = typename Traits::storage_type;
        using RuntimeType = typename Traits::storage_type;
        auto buffer = static_cast<StorageType *>(raw_buffer);

        for (int i = 0; i < dim_1; ++i) {
            for (int j = 0; j < dim_2; ++j) {
                buffer[j + i * dim_1] =
                    static_cast<StorageType>(RuntimeType(generator(i, j)));
            }
        }
    };

    invoke_with_traits(data_type, fill);
}

/// Checks that the output 2D matrix calculated by the shader is contains the
/// same values as runtime matmul of matrices with values defined by |lhs| and
/// |rhs|.
template<DataType OutputType, DataType InputType, typename Generator1Fn,
         typename Generator2Fn>
static void check_output(const ShaderCode &shader, void *raw_buffer,
                         unsigned M, unsigned N, unsigned K,
                         Generator1Fn lhs, Generator2Fn rhs) {
    using OutputTraits = DataTypeTraits<OutputType>;
    using OutputStorageType = typename OutputTraits::storage_type;
    using OutputRuntimeType = typename OutputTraits::runtime_type;
    using InputTraits = DataTypeTraits<InputType>;
    using InputRuntimeType = typename InputTraits::runtime_type;

    auto output = static_cast<OutputStorageType *>(raw_buffer);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            OutputRuntimeType acc(0.0f);
            for (int k = 0; k < K; ++k) {
                acc += OutputRuntimeType(InputRuntimeType(lhs(i, k))) *
                       OutputRuntimeType(InputRuntimeType(rhs(j, k)));
            }

            OutputRuntimeType gpuValue(output[i * N + j]);
            BM_CHECK_EQ(gpuValue, acc)
                << fmt::format("destination buffer element ({},{}) has incorrect value: expected to be {} but found {}\n\t^ In shader: {}, {}->{}",
                               i, j, acc, gpuValue, shader.name, get_name(shader.input_type), get_name(shader.output_type));
        }
    }
}

// Returns true iff |a| is a multiple of |b|.
static bool is_multiple_of(int a, int b) { return a >= b && a % b == 0; }

namespace luisa {
static void Mmt(::benchmark::State &state,
                LatencyMeasureMode mode,
                Device *device,
                const ShaderCode &shader, int M, int N, int K) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    DataType input_type = shader.input_type;
    DataType output_type = shader.output_type;
    const size_t src0_size = M * K * get_size(input_type);
    const size_t src1_size = K * N * get_size(input_type);
    const size_t dst_size = M * N * get_size(output_type);

    auto src0_buffer = device->create_buffer<float>(M * K);
    auto src1_buffer = device->create_buffer<float>(K * N);
    auto dst_buffer = device->create_buffer<float>(M * N);
    auto command = metal::MetalCommand::matmul(src0_buffer.view(), src1_buffer.view(), dst_buffer.view(),
                                               shader.M0, shader.N0, shader.K0,
                                               M, N, K,
                                               shader.wg_size_x, shader.wg_size_y);
    command->alloc_pso(device);

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    auto getLhs = [K](int i, int j) {
        float v = ((float)((i * K + j) % 5) - 1.0f) / 2.0f;
        return v;
    };
    auto getRhs = [K](int i, int j) {
        float v = ((float)((i * K + j) % 7) - 1.0f) / 2.0f;
        return v;
    };

    auto ptr = malloc(src0_size);
    fill_buffer(input_type, ptr, M, K, getLhs);
    stream << src0_buffer.copy_from(ptr) << synchronize();
    free(ptr);

    // In mmt, the RHS is input is transposed, which makes the matrix colum-major.
    ptr = malloc(src1_size);
    fill_buffer(input_type, ptr, N, K, getRhs);
    stream << src1_buffer.copy_from(ptr) << synchronize();
    free(ptr);

    //===-------------------------------------------------------------------===/
    // Clear the output buffer data set by the previous benchmark run
    //===-------------------------------------------------------------------===/

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/
    {
        stream << command->clone()
               << synchronize();
    }

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/
    if (input_type == DataType::i8) {
        ptr = malloc(dst_size);
        stream << dst_buffer.copy_to(ptr) << synchronize();
        check_output<DataType::i32, DataType::i8>(shader, ptr, M,
                                                  N, K, getLhs, getRhs);
        free(ptr);
    }

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/
    for ([[maybe_unused]] auto _ : state) {
        auto start_time = std::chrono::high_resolution_clock::now();
        stream << command->clone()
               << synchronize();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        switch (mode) {
            case LatencyMeasureMode::kSystemSubmit:
                state.SetIterationTime(elapsed_seconds.count());
                break;
        }
    }

    double numOperation = double(N) * double(M) * double(K) * 2.;
    state.counters["FLOps"] =
        ::benchmark::Counter(numOperation,
                             ::benchmark::Counter::kIsIterationInvariant |
                                 ::benchmark::Counter::kIsRate,
                             ::benchmark::Counter::kIs1000);
}

void MMT::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    for (const ShaderCode &shader : kShaderCodeCases) {
        std::string matmul_size = fmt::format("{}x{}x{}", M, N, K);
        std::string tiling_scheme = fmt::format("{}x{}x{}", shader.M0, shader.N0, shader.K0);
        BM_CHECK(is_multiple_of(M, shader.M0))
            << fmt::format("Incompatible tiling scheme: {}", tiling_scheme);
        BM_CHECK(is_multiple_of(N, shader.N0))
            << fmt::format("Incompatible tiling scheme: {}", tiling_scheme);
        BM_CHECK(is_multiple_of(K, shader.K0))
            << fmt::format("Incompatible tiling scheme: {}", tiling_scheme);
        BM_CHECK(is_multiple_of(shader.K0, 4))
            << fmt::format("Incompatible tiling scheme: {}", tiling_scheme);

        std::string workgroup_size = fmt::format("{}x{}x1", shader.wg_size_x, shader.wg_size_y);
        std::string type_info = fmt::format("{}->{}", get_name(shader.input_type), get_name(shader.output_type));
        std::string test_name = fmt::format("{}/mmt[{}]/{}/{}/Workgroup[{}]", gpu_name, matmul_size, type_info, shader.name, workgroup_size);

        ::benchmark::RegisterBenchmark(test_name, Mmt, mode, &device,
                                       shader, M, N, K)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
    }
}

}// namespace luisa