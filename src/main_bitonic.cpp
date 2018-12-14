#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, const std::string &message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    // Device
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    // Data
    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // CPU
    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << double(n) / 1000 / 1000 / t.lapAvg() << " millions/s" << std::endl;
    }

    // GPU
    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    {
        ocl::Kernel bitonic_local(bitonic_kernel, bitonic_kernel_length, "bitonic_local");
        bitonic_local.compile();
        ocl::Kernel bitonic_initial(bitonic_kernel, bitonic_kernel_length, "bitonic_initial");
        bitonic_initial.compile();

        unsigned int wg_size = 256;
        unsigned int gw_size = (n + wg_size - 1) / wg_size * wg_size;
        unsigned int max_k = 2;
        while (max_k < n) max_k *= 2;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();

            unsigned int k = 2;
            while (k <= max_k) {
                auto cur_k = k / 2;

                while (2 * cur_k > wg_size) {
                    bitonic_initial.exec(gpu::WorkSize(wg_size, gw_size), as_gpu, cur_k, k, n);
                    cur_k /= 2;
                }

                bitonic_local.exec(gpu::WorkSize(wg_size, gw_size), as_gpu, cur_k, k, n);
                k *= 2;
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << double(n) / 1000 / 1000 / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Test
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
