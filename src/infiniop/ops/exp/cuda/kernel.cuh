#ifndef __EXP_CUDA_H__
#define __EXP_CUDA_H__

namespace op::exp::cuda {
typedef struct ExpOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>

    __global__ void exp_kernel(float* out, const float* in, int n) {
        extern __shared__ float buffer[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            buffer[threadIdx.x] = in[idx];  // 加载到共享内存
            __syncthreads();
            out[idx] = __expf(buffer[threadIdx.x]);  // 计算并写回
        }
    }
} ExpOp;
} // namespace op::exp::cuda

#endif // __EXP_CUDA_H__