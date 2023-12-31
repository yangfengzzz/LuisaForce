//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "utils/data_type_util.h"
#include "utils/status_util.h"
#include "runtime/buffer.h"
#include "runtime/stream.h"
#ifdef LUISA_PLATFORM_APPLE
#include "runtime/ext/metal/metal_command.h"
#include "runtime/ext/metal/mps_command.h"
#include "runtime/ext/debug_capture_ext.h"
#endif

#ifdef LUISA_PLATFORM_CUDA
#include "runtime/ext/cuda/cuda_commands.h"
#endif

#include <spdlog/fmt/fmt.h>

#include "benchmark_api.h"

struct ShaderCodeBase {
    const char *name;    // Shader case name
    DataType input_type; // LHS & RHS element type
    DataType output_type;// Output/Result matrix element type

    ShaderCodeBase(const char *name, DataType input_type, DataType output_type)
        : name{name},
          input_type{input_type},
          output_type{output_type} {}

    [[nodiscard]] virtual uint tileM() const = 0;
    [[nodiscard]] virtual uint tileN() const = 0;
    [[nodiscard]] virtual uint tileK() const = 0;
    [[nodiscard]] virtual uint wg_size_x() const = 0;
    [[nodiscard]] virtual uint wg_size_y() const = 0;

#ifdef LUISA_PLATFORM_APPLE
    virtual luisa::compute::metal::MetalCommand::UCommand create_command(Device *device,
                                                                         BufferView<float> src0_buffer,
                                                                         BufferView<float> src1_buffer,
                                                                         BufferView<float> dst_buffer,
                                                                         uint M, uint N, uint K) const = 0;
#endif

#ifdef LUISA_PLATFORM_CUDA
    virtual luisa::compute::cuda::CudaCommand::UCommand create_command(Device *device,
                                                                       BufferView<float> src0_buffer,
                                                                       BufferView<float> src1_buffer,
                                                                       BufferView<float> dst_buffer,
                                                                       uint M, uint N, uint K) const = 0;
#endif
};

template<uint TILE_M, uint TILE_N, uint TILE_K, uint WG_X, uint WG_Y>
struct ShaderCode : ShaderCodeBase {
    ShaderCode(const char *name, DataType input_type, DataType output_type)
        : ShaderCodeBase{name, input_type, output_type} {
    }

    [[nodiscard]] uint tileM() const override {
        return TILE_M;
    }

    [[nodiscard]] uint tileN() const override {
        return TILE_N;
    }

    [[nodiscard]] uint tileK() const override {
        return TILE_K;
    }

    [[nodiscard]] uint wg_size_x() const override {
        return WG_X;
    }

    [[nodiscard]] uint wg_size_y() const override {
        return WG_Y;
    }

#ifdef LUISA_PLATFORM_APPLE
    luisa::compute::metal::MetalCommand::UCommand create_command(Device *device,
                                                                 BufferView<float> src0_buffer,
                                                                 BufferView<float> src1_buffer,
                                                                 BufferView<float> dst_buffer,
                                                                 uint M, uint N, uint K) const override {
        auto command = metal::MetalCommand::matmul(src0_buffer, src1_buffer, dst_buffer,
                                                   tileM(), tileN(), tileK(),
                                                   M, N, K,
                                                   wg_size_x(), wg_size_y());
        command->alloc_pso(device);

        auto mps_command = metal::MPSCommand::gemm(src0_buffer, src1_buffer, dst_buffer,
                                                   M, N, K);
        return command;
    }
#endif

#ifdef LUISA_PLATFORM_CUDA
    luisa::compute::cuda::CudaCommand::UCommand create_command(Device *device,
                                                               BufferView<float> src0_buffer,
                                                               BufferView<float> src1_buffer,
                                                               BufferView<float> dst_buffer,
                                                               uint M, uint N, uint K) const override {
        return cuda::CudaCommand::matmul<TILE_M, TILE_N, TILE_K, WG_X, WG_Y>(src0_buffer, src1_buffer, dst_buffer,
                                                                             M, N, K);
    }
#endif
};

#define SHADER_TILE_F32(M, N, K, X, Y)           \
    std::make_unique<ShaderCode<M, N, K, X, Y>>( \
        "Tile[" #M "x" #N "x" #K "]",            \
        DataType::fp32, DataType::fp32)

#define SHADER_TILE_I32(M, N, K, X, Y)           \
    std::make_unique<ShaderCode<M, N, K, X, Y>>( \
        "Tile[" #M "x" #N "x" #K "]",            \
        DataType::i32, DataType::i32)

#define WORKGROUP_TILE_N_F32(X, Y, N)                                    \
    SHADER_TILE_F32(2, N, 4, X, Y), SHADER_TILE_F32(4, N, 4, X, Y),      \
        SHADER_TILE_F32(8, N, 4, X, Y), SHADER_TILE_F32(16, N, 4, X, Y), \
        SHADER_TILE_F32(32, N, 4, X, Y), SHADER_TILE_F32(2, N, 8, X, Y), \
        SHADER_TILE_F32(4, N, 8, X, Y), SHADER_TILE_F32(8, N, 8, X, Y),  \
        SHADER_TILE_F32(16, N, 8, X, Y), SHADER_TILE_F32(32, N, 8, X, Y)

#define WORKGROUP_TILE_N_I32(X, Y, N)                                    \
    SHADER_TILE_I32(2, N, 4, X, Y), SHADER_TILE_I32(4, N, 4, X, Y),      \
        SHADER_TILE_I32(8, N, 4, X, Y), SHADER_TILE_I32(16, N, 4, X, Y), \
        SHADER_TILE_I32(32, N, 4, X, Y), SHADER_TILE_I32(2, N, 8, X, Y), \
        SHADER_TILE_I32(4, N, 8, X, Y), SHADER_TILE_I32(8, N, 8, X, Y),  \
        SHADER_TILE_I32(16, N, 8, X, Y), SHADER_TILE_I32(32, N, 8, X, Y)

static std::unique_ptr<ShaderCodeBase> kShaderCodeCases[] = {
    WORKGROUP_TILE_N_F32(16, 1, 64),
    WORKGROUP_TILE_N_F32(16, 1, 128),
    WORKGROUP_TILE_N_I32(16, 1, 64),
    WORKGROUP_TILE_N_I32(16, 1, 128),
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
static void check_output(const ShaderCodeBase *shader, void *raw_buffer,
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
                       OutputRuntimeType(InputRuntimeType(rhs(k, j)));
            }

            OutputRuntimeType gpuValue(output[i * N + j]);
            BM_CHECK_EQ(gpuValue, acc) << fmt::format("destination buffer element ({},{}) has incorrect value: "
                                                      "expected to be {} but found {}\n\t^ In shader: {}, {}->{}",
                                                      i, j, acc, gpuValue, shader->name,
                                                      get_name(shader->input_type), get_name(shader->output_type));
        }
    }
}

namespace luisa {
static void matmul(::benchmark::State &state,
                   LatencyMeasureMode mode,
                   Device *device,
                   const ShaderCodeBase *shader, int M, int N, int K) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    DataType input_type = shader->input_type;
    DataType output_type = shader->output_type;
    const size_t src0_size = M * K * get_size(input_type);
    const size_t src1_size = K * N * get_size(input_type);
    const size_t dst_size = M * N * get_size(output_type);

    auto src0_buffer = device->create_buffer<float>(M * K);
    auto src1_buffer = device->create_buffer<float>(K * N);
    auto dst_buffer = device->create_buffer<float>(M * N);
    auto command = shader->create_command(device,
                                          src0_buffer.view(), src1_buffer.view(), dst_buffer.view(),
                                          M, N, K);

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    auto getSrc0 = [K](int i, int j) {
        float v = ((float)((i + j * K) % 5) - 1.0f) / 2.0f;
        return v;
    };
    auto getSrc1 = [N](int i, int j) {
        float v = ((float)((i + j * N) % 7) - 1.0f) / 2.0f;
        return v;
    };

    auto ptr = malloc(src0_size);
    fill_buffer(input_type, ptr, M, K, getSrc0);
    stream << src0_buffer.copy_from(ptr) << synchronize();
    free(ptr);

    ptr = malloc(src1_size);
    fill_buffer(input_type, ptr, K, N, getSrc1);
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
    if (output_type == DataType::fp32) {
        ptr = malloc(dst_size);
        stream << dst_buffer.copy_to(ptr) << synchronize();
        check_output<DataType::fp32, DataType::fp32>(shader, ptr, M, N, K,
                                                     getSrc0, getSrc1);
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

void MatMul::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    for (DataType input_type : {DataType::fp32}) {
        for (const auto &shader : kShaderCodeCases) {
            if (shader->input_type != input_type) continue;
            int paddM = (M + shader->tileM() - 1) / shader->tileM() * shader->tileM();
            int paddN = (N + shader->tileN() - 1) / shader->tileN() * shader->tileN();
            std::string matmul_size = fmt::format("{}x{}x{}", M, N, K);
            std::string workgroup_size = fmt::format("{}x{}x1", shader->wg_size_x(), shader->wg_size_y());
            std::string type_info = fmt::format("{}->{}", get_name(shader->input_type), get_name(shader->output_type));
            std::string test_name = fmt::format("{}/Matmul[{}]/{}/{}/Workgroup[{}]",
                                                gpu_name, matmul_size, type_info, shader->name, workgroup_size);

            ::benchmark::RegisterBenchmark(test_name, matmul, mode, &device,
                                           shader.get(), paddM, paddN, K)
                ->UseManualTime()
                ->Unit(::benchmark::kMicrosecond);
        }
    }
}

}// namespace luisa
