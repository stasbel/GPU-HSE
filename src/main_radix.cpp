#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/radix_cl.h"

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

gpu::gpu_mem_32u prefixSum(ocl::Kernel &scan, gpu::gpu_mem_32u &as_gpu, unsigned int n, unsigned int wg_size) {
    unsigned int res_n = (n + wg_size - 1) / wg_size * wg_size;

    gpu::gpu_mem_32u sums;
    sums.resizeN(res_n);
    gpu::gpu_mem_32u b_sums;
    b_sums.resizeN(res_n / wg_size + 1);

    unsigned int z = 0;
    b_sums.writeN(&z, 1);

    scan.exec(gpu::WorkSize(wg_size, n), as_gpu, b_sums, sums, n, 1);

    if (wg_size >= n)
        return sums;

    b_sums = prefixSum(scan, b_sums, res_n / wg_size + 1, wg_size);
    scan.exec(gpu::WorkSize(wg_size, n), as_gpu, b_sums, sums, n, 0);

    return sums;
}

int main(int argc, char **argv) {
    // Device
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    // Data
    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // CPU
    std::vector<unsigned int> cpu_sorted;
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
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    {
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();
        ocl::Kernel scan(radix_kernel, radix_kernel_length, "scan");
        scan.compile();
        ocl::Kernel reorder(radix_kernel, radix_kernel_length, "reorder");
        reorder.compile();

        unsigned int mask_width = 2, wg_size = 256, gw_size = n;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();

            gpu::gpu_mem_32u counts_gpu;
            counts_gpu.resizeN(n * (1 << mask_width));
            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(n);

            for (int i = 0; i < 32; i += mask_width) {
                count.exec(gpu::WorkSize(wg_size, gw_size), as_gpu, counts_gpu, n, i, mask_width);
                gpu::gpu_mem_32u s = prefixSum(scan, counts_gpu, n * (1 << mask_width), wg_size);
                reorder.exec(gpu::WorkSize(wg_size, gw_size), as_gpu, s, bs_gpu, n, i, mask_width);
                as_gpu.swap(bs_gpu);
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
