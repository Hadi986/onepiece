## 查看cuda版本
torch.cuda.is_available() # 查看cuda是否可用
torch.version.cuda # 查看安装pytorch时安装的cuda版本，但并不一定是实际使用的版本，
import torch.utils.cpp_extension
torch.utils.cpp_extension.CUDA_HOME   # 查看实际使用的cuda版本，若不是为了看cuda的路径也可以使用nvcc -V 命令快速查看

## 更改cuda
export PATH = "cuda_path/bin:$PATH"
export PATH = "cuda_path/lib64:$PATH"





