#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define WORK_GROUP_SIZE 256

#line 10

__kernel void count(
        __global unsigned int *as, __global unsigned int *cs,
        unsigned int n, unsigned int mask_offset, unsigned int mask_width
) {
    unsigned int gid = get_global_id(0), mask = ((1UL << mask_width) - 1) << mask_offset, n_vals = 1 << mask_width;

    if (gid < n) {
        for (int i = 0; i < n_vals; i++)
            cs[i * n + gid] = ((as[gid] & mask) >> mask_offset) == i ? 1 : 0;
    }
}

__kernel void scan(
        __global unsigned int *as, __global unsigned int *bs, __global unsigned int *cs,
        unsigned int n, int zero_bs
) {
    unsigned int gid = get_global_id(0), lid = get_local_id(0), block_size = get_local_size(0), gr_id = get_group_id(0);

    __local unsigned int local_a[WORK_GROUP_SIZE], local_b[WORK_GROUP_SIZE];
    __local unsigned int *a = local_a, *b = local_b;

    if (gid < n) {
        a[lid] = as[gid];

        if (lid == 0) {
            if (!zero_bs)
                a[0] += bs[gr_id];
        }
    } else {
        a[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = 1; s < block_size; s <<= 1) {
        if (lid > (s - 1)) {
            b[lid] = a[lid] + a[lid - s];
        } else {
            b[lid] = a[lid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        __local unsigned int *tmp = a;
        a = b;
        b = tmp;
    }

    if (gid < n) {
        cs[gid] = a[lid];

        if (lid == 0)
            bs[gr_id + 1] = a[block_size - 1];
    }
}

__kernel void reorder(
        __global unsigned int *as, __global unsigned int *os, __global unsigned int *bs,
        unsigned int n, unsigned int mask_offset, unsigned int mask_width
) {
    unsigned int gid = get_global_id(0), mask = ((1UL << mask_width) - 1) << mask_offset, n_vals = 1 << mask_width;

    if (gid < n) {
        bs[os[((as[gid] & mask) >> mask_offset) * n + gid] - 1] = as[gid];
    }
}