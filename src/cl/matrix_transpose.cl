#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 8

__kernel void matrix_transpose(__global float *a, __global float *r, int h, int w, __local float *la) {
    const int gi = get_global_id(1), gj = get_global_id(0);
    const int gh = get_global_size(1), gw = get_global_size(0);
    const int li = get_local_id(1), lj = get_local_id(0);
    const int ls = get_local_size(0);
    const int ki = get_group_id(1), kj = get_group_id(0);

    if (gi < h && gj < w) la[li * ls + lj] = a[gi * gw + gj];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gi < h && gj < w) r[(kj * ls + li) * gw + (ki * ls + lj)] = la[lj * ls + li];
}