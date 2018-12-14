#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define WORK_GROUP_SIZE 256

#line 10

__kernel void count(
        __global unsigned int *as, __global unsigned int *cs,
        unsigned int n, unsigned int mask_offset, unsigned int mask_width
) {
    unsigned int g_id = get_global_id(0), mask = ((1UL << mask_width) - 1) << mask_offset;

    if (g_id < n) {
        for (int i = 0; i < (1 << mask_width); i++) {
            cs[i * n + g_id] = ((as[g_id] & mask) >> mask_offset) == i ? 1 : 0;
        }
    }
}

__kernel void scan(
        __global unsigned int *as, __global unsigned int *bs, __global unsigned int *cs,
        unsigned int n, int zero_bs
) {
    unsigned int g_id = get_global_id(0), l_id = get_local_id(0), wg_size = get_local_size(0), gr_id = get_group_id(0);

    __local unsigned int las[WORK_GROUP_SIZE], lbs[WORK_GROUP_SIZE];
    __local unsigned int *a = las, *b = lbs;

    if (g_id < n) {
        a[l_id] = as[g_id];

        if (l_id == 0) {
            if (!zero_bs) {
                a[0] += bs[gr_id];
            }
        }
    } else {
        a[l_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 1; s < wg_size; s <<= 1) {
        if (l_id > (s - 1)) {
            b[l_id] = a[l_id] + a[l_id - s];
        } else {
            b[l_id] = a[l_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local unsigned int *tmp = a;
        a = b;
        b = tmp;
    }

    if (g_id < n) {
        cs[g_id] = a[l_id];

        if (l_id == 0) {
            bs[gr_id + 1] = a[wg_size - 1];
        }
    }
}

__kernel void reorder(
        __global unsigned int *as, __global unsigned int *os, __global unsigned int *bs,
        unsigned int n, unsigned int mask_offset, unsigned int mask_width
) {
    unsigned int g_id = get_global_id(0), mask = ((1UL << mask_width) - 1) << mask_offset;

    if (g_id < n) {
        bs[os[((as[g_id] & mask) >> mask_offset) * n + g_id] - 1] = as[g_id];
    }
}