#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

__kernel void sum(__global const unsigned int *as,
                  __global unsigned int *res,
                  unsigned int n) {
    const size_t localId = get_local_id(0), globalId = get_global_id(0);

    __local unsigned localAs[WORK_GROUP_SIZE];
    if (globalId >= n)
        return;
    localAs[localId] = as[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned valuesToSum = WORK_GROUP_SIZE; valuesToSum > 1; valuesToSum /= 2) {
        if (2 * localId < valuesToSum) {
            localAs[localId] += localAs[localId + valuesToSum / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) atomic_add(res, localAs[0]);
}
