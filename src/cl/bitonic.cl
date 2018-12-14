#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define WORK_GROUP_SIZE 256

#line 10

__kernel void bitonic_local(__global float *as, unsigned int cur_k, unsigned int k, unsigned int n) {
    unsigned int global_id = get_global_id(0), local_id = get_local_id(0);

    __local float mem[WORK_GROUP_SIZE];
    if (global_id < n) {
        mem[local_id] = as[global_id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int is_sort_ascending = global_id % (2 * k) < k;
    while (cur_k >= 1) {
        if (global_id % (2 * cur_k) < cur_k && global_id + cur_k < n) {
            float a = mem[local_id], b = mem[local_id + cur_k];

            if ((a < b) != is_sort_ascending) {
                mem[local_id] = b;
                mem[local_id + cur_k] = a;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        cur_k /= 2;
    }

    if (global_id < n) {
        as[global_id] = mem[local_id];
    }
}


__kernel void bitonic_initial(__global float *as, unsigned int cur_k, unsigned int k, unsigned int n) {
    unsigned int global_id = get_global_id(0);

    if (global_id % (2 * cur_k) < cur_k && global_id + cur_k < n) {
        float a = as[global_id], b = as[global_id + cur_k];

        if ((a < b) != (global_id % (2 * k) < k)) {
            as[global_id] = b;
            as[global_id + cur_k] = a;
        }
    }
}