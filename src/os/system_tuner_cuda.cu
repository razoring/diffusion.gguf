#include "system_tuner.h"

#if defined(GGML_USE_CUDA) || defined(DIFFUSION_USE_CUDA)
#include <cuda_runtime.h>

namespace diffusion {
namespace os {

void query_cuda_specs(SystemProfile& profile) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count <= 0) {
        profile.gpu_count = 0;
        return;
    }
    profile.gpu_count = count;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (i == 0) {
                profile.primary_gpu_name = prop.name;
            }
            profile.total_vram_bytes += prop.totalGlobalMem;
        }
        size_t free_b = 0, total_b = 0;
        cudaSetDevice(i);
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            profile.free_vram_bytes += free_b;
        }
    }
    cudaSetDevice(0);
}

} // namespace os
} // namespace diffusion
#else
namespace diffusion {
namespace os {
void query_cuda_specs(SystemProfile& profile) {
    profile.gpu_count = 0;
}
} // namespace os
} // namespace diffusion
#endif
