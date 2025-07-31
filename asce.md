# venus
~/sfyan/ivse/prebuilt/Magik/InferenceKit/venus/mips/public/7.2.0/2.29/lib/uclibc
~/sfyan/ivse/prebuilt/Magik/InferenceKit/venus/mips/internal/7.2.0/2.29/lib/uclibc
T41、A1需要api.so和drivers.so
cp api.so drivers.so mips/public/7.2.0/2.29/lib/uclibc

## checkout branch
current-path:~/sfyan/ivse/prebuilt
```
if T40 or X2500
git checkout agt
if T41 or A1
git checkout nna2
```
# MagiKTrainingKit
```
git clone ssh://cywang@10.3.2.212:29418/Magik/MagiKTrainingKit
git checkout FX
git pull
current-path:MagiKTrainingKit
检查CMakelist.txt,修改SMS为显卡支持的算力，如高性能显卡删除30，增添86
set(SMS  **30**;37;50;52;60;61)
###cuda version == 9
if(${CUDA_VERSION_MAJOR} EQUAL 9)
  set(SMS ${SMS};70)
endif()
###cuda version >=10
if(${CUDA_VERSION_MAJOR} GREATER 9)
  set(SMS ${SMS};70;75;**86**)
endif()

source env/env_setup.sh
mkdir build
cd build
cmake ..
make -j
```
如果,寻找机子当前环境cuda版本一致的，因为插件编译时默认先选择机器的cuda
```
export PATH="/usr/local/cuda-X/bin:"
export LD_LIBRARY_PATH="/usr/local/cuda-X/lib64"
```
如果python版本与当前环境不一致
```
export PATH="home/user/miniconda3/envs/openmmlab/bin:"
```
# TransformKit
## 编译
准备工作：
```console
cd /data_bak/sfyan/ivse(branch:agt)
sourece build/env.sh
cd ivse/prebuilt
git pull
```
修改Cmakelits.txt
```console
# line 67-79
exec_program(protoc ARGS "--version 2>&1 | sed -n '/libprotoc/p' | awk -F ' ' '{print $2}'" OUTPUT_VARIABLE PROTO_VERSION RETURN_VALUE CMD_RETURN)
if(${PROTO_VERSION} STRLESS "3.6.0")
  message(FATAL_ERROR "protoc version must great 3.6.0.")
endif()
......
set(ENV{PATH} "$ENV{IVSE_ENV_PREBUILT_THIRD_PARTY_DIR}/protobuf/x86_64/${GCC_VERSION}/bin/:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{IVSE_ENV_PREBUILT_THIRD_PARTY_DIR}/protobuf/x86_64/${GCC_VERSION}/lib/:$ENV{LD_LIBRARY_PATH}")
===========>
set(ENV{PATH} "$ENV{IVSE_ENV_PREBUILT_THIRD_PARTY_DIR}/protobuf/x86_64/${GCC_VERSION}/bin/:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{IVSE_ENV_PREBUILT_THIRD_PARTY_DIR}/protobuf/x86_64/${GCC_VERSION}/lib/:$ENV{LD_LIBRARY_PATH}")

exec_program(protoc ARGS "--version 2>&1 | sed -n '/libprotoc/p' | awk -F ' ' '{print $2}'" OUTPUT_VARIABLE PROTO_VERSION RETURN_VALUE CMD_RETURN)
if(${PROTO_VERSION} STRLESS "3.6.0")
  message(FATAL_ERROR "protoc version must great 3.6.0.")
endif()
......
```
```console
cd MagikTransformKit/build
cmake ..
make -j
```
## 使用
```
生成Log文件和打印中间过程
export MAGIK_CPP_MIN_LOG_LEVEL=0
生成内部模型
1.export MAGIK_CPP_FOR_INTERNAL=true
生成的模型在xxxx-xx-xx-magik/model_xxx_t40/xx.bin
2.InferenceKit要使用internal,
e.g.:
# Makefile TOPDIR ?= /data_bak/hyu/git_magik-toolkit/InferenceKit/nna1_inter/mips720-glibc229/
or TOPDIR ?= ivse/prebuilt/Magik/InferenceKit/venus/mips/internal/7.2.0/2.29/
3.inference.cpp
#include "venus.h"==>include "venus_api.h"
#net_create ==> predictor_create
...
make时会报错，提示函数名变化。
# set the save_path of result(input_data.bin)
MAGIK_TRAININGKIT_DUMP=1 
MAGIK_TRAININGKIT_PATH='/data_bak/cywang/tmp' 
make build_type=release/profile/nmem/debug clean
```
# 数据校对
```
if python_data_md5sum != slug_data_md5sum
to compare py.bin and slug.bin
$ python compare_bin.py
$ python compare_bin_data.py
```
# 文件比较工具
```
meld a.txt b.txt c.txt
meld a b
```

# 上板
TOPDIR ?= /data_bak/sfyan/ivse/prebuilt/Magik/InferenceKit/venus/mips/T41/
CROSS_COMPILE:= /data_bak/mobach/ivse/mips-linux-gnu-ingenic-gcc7.2.0-glibc2.29-fp64-r5.1.7/bin/mips-linux-gnu-

mem ==> system
rmem ==> video
nmem ==> image(venus)
View storage:
```
$ cat /proc/cmdlines
```
# Trainingkit dump data
MAGIK_TRAININGKIT_DEVP_DUMP=1 
MAGIK_TRAININGKIT_PATH="./tmp"

# ivse、MagikTraining、magik-toolkit、git_magik-toolkit
magik-toolkit ==> public
git_magik-toolkit ==> internal
MagikTraining ==> develop(source code)
ivse ==> magik1.0

# mount PC to slug
```
/system/init/app_init.sh
ifconfig eth0 10.1.10.66
route add default gw 10.1.10.1
```
## passward 
```
10.3.2.232: ivse
10.3.2.212: FxModelZoo
http://10.3.2.212:8082/#/q/status:open
cywang cywang123
RDM: chunyu.wang Jz1234567
Citrix: st\cywang cyw@920
e-mail: ace.cywang Wwwwangyu@147
```

RDM提交文档：
主页面==>文档仓库==>新建（按流程一步步走）*都要填写，最下面没有 *的所属部门也必须要填

# 查看进程
ps -aux | grep python

# 去除模型后处理
1.在forward中，去除后处理，生成onnx
2.在onnx中移除后处理（繁琐）
<!-- 多个输出 -->
```console
def clip_model(onnx_model_path, output_node_name):
    import onnx
    onnx_file = onnx_model_path
    save = "delete_detect.onnx"
    model = onnx.load_model(onnx_file)

    node = model.graph.node

    index = []
    Conv_248_index = 1
    Conv_298_index = 100000
    Conv_198_index = 100000
    start_index = 100009
    for i in range(len(node)):
        #print(node[i])
        for tmp in node[i].output:
            #print(tmp)
            if tmp == "1596":
                print('================',i)
                Conv_248_index = i
            elif tmp == "1223":
                Conv_298_index = i
            elif tmp == "1199":
                s1 = i
                print('s1',i)
            elif tmp == "1572":
                s2 = i
                print('s2',i)
            elif tmp == "2118":
                s3 = i
                print('s3',i)
            elif tmp == "2139":
                s4 = i
                print('s4',i)
            elif  tmp == "850":
                Conv_198_index = i
                start_index = i
    for i in range(len(node)):
        if i > start_index:
            if i < s3 or (i > Conv_298_index and i < s4 ) or (i > Conv_248_index):
                #print(Conv_248_index)
                #print(i)
                index.append(i)
    print(Conv_248_index, Conv_298_index,Conv_198_index)
    #exit()
    for i in reversed(index):
        node.remove(node[i])
    out = model.graph.output
    del out[0]
    
    out[0].name = "1596"
    out[0].type.tensor_type.shape.dim[0].dim_value = 1
    out[0].type.tensor_type.shape.dim[1].dim_value = 80
    out[0].type.tensor_type.shape.dim[2].dim_value = 80
    out[0].type.tensor_type.shape.dim[3].dim_value = 255

    del out[0].type.tensor_type.shape.dim[4]
    

    out[1].name = "1223"
    out[1].type.tensor_type.shape.dim[0].dim_value = 1
    out[1].type.tensor_type.shape.dim[1].dim_value = 40
    out[1].type.tensor_type.shape.dim[2].dim_value = 40
    out[1].type.tensor_type.shape.dim[3].dim_value = 255
    del out[1].type.tensor_type.shape.dim[4]
    
    out[2].name = "850"
    #print(out)
    out[2].type.tensor_type.shape.dim[0].dim_value = 1
    out[2].type.tensor_type.shape.dim[1].dim_value = 20
    out[2].type.tensor_type.shape.dim[2].dim_value = 20
    out[2].type.tensor_type.shape.dim[3].dim_value = 255

    del out[2].type.tensor_type.shape.dim[4]

    #print(out[0].type.tensor_type.shape)
   
    #print('node',model.graph.node)
    onnx.save(model, save)
    onnx.checker.check_model(model)
<!-- 单个输出 -->
def clip_model1(onnx_model_path, output_node_name):
    onnx_model = onnx.load_model(onnx_model_path)
    # print(onnx_model.graph.output)
    new_nodes = []
    for node in onnx_model.graph.node:
        is_stop = False
        for tmp in node.input:
            if tmp == output_node_name:
                is_stop = True
        if is_stop:
            break
        else:
            new_nodes.append(node)
    del onnx_model.graph.node[:]
    del onnx_model.graph.output[1:]
    onnx_model.graph.output[0].name = "output"
    for i in range(len(new_nodes)):
        if new_nodes[i].output[0] == output_node_name:
            # new_nodes[i].output[0] = "res_net/dense/Softmax:0"
            new_nodes[i].output[0] = "output"
    onnx_model.graph.node.extend(new_nodes)
    with open("./clip_onnx_model.onnx", "wb") as f:
        onnx.save_model(onnx_model, f)
    return "./clip_onnx_model.onnx"
```