#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY,
                         const unsigned int iters) {

    const float threshold = 256.0f;
    const size_t i = get_global_id(0), j = get_global_id(1);
    const size_t width = get_global_size(0), height = get_global_size(1);

    float x = fromX + sizeX * ((i + 0.5f) / width),
            y = fromY + sizeY * ((j + 0.5f) / height);
    float xPrev, x0 = x, y0 = y;
    unsigned int iter = 0;
    for (; iter < iters; iter++) {
        xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if (x * x + y * y > threshold * threshold) {
            break;
        }
    }

    results[j * width + i] = (float) iter / iters;
}
