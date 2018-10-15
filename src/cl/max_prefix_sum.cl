#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 8

#define WORK_GROUP_SIZE 64

__kernel void max_prefix_sum(__global int *ns, __global int *ms, __global int *is,
                             int n,
                             __local int *lns, __local int *lms, __local int *lis) {
    const int i = get_global_id(0), li = get_local_id(0), ls = get_local_size(0);

    lns[li] = 0;
    lms[li] = 0;
    lis[li] = 0;

    if (i < n) {
        lns[li] = ns[i];
        lms[li] = ms[i];
        lis[li] = is[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = 1; j < ls; j *= 2) {
        int mask = (j - 1) | 1;

        if ((li & mask) == 0 && li + j < ls && i + j < n) {
            lis[li] = lms[li] > lns[li] + lms[li + j] ? lis[li] : lis[li + j];
            lms[li] = max(lms[li], lns[li] + lms[li + j]);
            lns[li] += lns[li + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    }

    if (li == 0) {
        ns[i] = lns[0];
        ms[i] = lms[0];
        is[i] = lis[0];
    }
}