#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 8

__kernel void matrix_multiplication(__global float *a, __global float *b, __global float *c,
                                    int k, int m, int n,
                                    __local float *la, __local float *lb, __local float *lc) {
    const int gi = get_global_id(1), gj = get_global_id(0);
    const int gh = get_global_size(1), gw = get_global_size(0);
    const int li = get_local_id(1), lj = get_local_id(0);
    const int ls = get_local_size(0);
    const int ki = get_group_id(1), kj = get_group_id(0);

    lc[li * ls + lj] = 0;

    for (int i = 0; i < get_num_groups(0); i++) {
        la[li * ls + lj] = a[(ki * ls + li) * gw + (i * ls + lj)];
        lb[li * ls + lj] = b[(i * ls + li) * gw + (kj * ls + lj)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < ls; j++) lc[li * ls + lj] += la[li * ls + j] * lb[j * ls + lj];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    c[gi * gw + gj] = lc[li * ls + lj];
}