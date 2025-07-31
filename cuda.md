## 查看cuda版本
torch.cuda.is_available() # 查看cuda是否可用
torch.version.cuda # 查看安装pytorch时安装的cuda版本，但并不一定是实际使用的版本，
import torch.utils.cpp_extension
torch.utils.cpp_extension.CUDA_HOME   # 查看实际使用的cuda版本，若不是为了看cuda的路径也可以使用nvcc -V 命令快速查看

## 更改cuda
export PATH = "cuda_path/bin:$PATH"
export PATH = "cuda_path/lib64:$PATH"

## CUDA编程
CUDA 的设计思想大致是：向显卡提交一个又一个任务，每一个任务都形如 “给定一个函数，与调用它的参数，请在显卡上运行这个函数”。我们一般称这种 “在显卡上运行的函数” 叫做 CUDA Kernel。仔细想想，这种设计很合理嘛！毕竟现在 GPU 是 “加速器”，其仅负责加速程序中的某一些部分，其他的控制流程与计算还是要由 CPU 来做的。

- 如何定义（创建）一个 CUDA Kernel？
    首先是如何定义 CUDA Kernel 的问题。CUDA C++ 中有三类函数：
    __host__: 这类函数与正常的函数没有区别。其只能被 host 上执行的函数（__host__）调用，并在 host 上执行。
    __global__: 这类函数可以被任何函数调用，并在 device 上执行。
    __device__: 这类函数只能被 device 上执行的函数（__device__ 或 __global__）调用，并在 device 上执行。
注：在 CUDA 的编程模型中，一般称 CPU 为 Host，GPU 为 Device。

- 如何调用这个 CUDA Kernel？
    那么，如何调用 CUDA Kernel 呢？与 C++ 中调用函数的方式大同小异，不过要在函数名与参数列表的中间加上一个 <<<GRID_DIM, BLOCK_DIM>>>（现阶段，先认为 GRID_DIM 与 BLOCK_DIM 均为 1）。举个例子：

    // 下面这句话调用了一个名为 gemm_gpu_1thread_kernel 的 kernel
    gemm_gpu_1thread_kernel<<<1, 1>>>(C, A, B, n, m, k);


