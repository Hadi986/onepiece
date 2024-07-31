# modelzo model

## 1.DPCRN
### DPCRN数据核对技巧
跑通完整的一个音频dump数据需要较长时间，因为需要跑一整个numblock，一个numblock有几百次，而每次运行模型都会dump一次数据，所以可以只跑第一次的。
跑一次forward如下：
python infer推理跑一次后直接退出即可，板端也是inference.cpp跑一次退出，重点在于如何获取第一次输入，因为dump出的input不对，

## 2.yolov8-seg
1.export onnx要将detect的forward处

2.resume_train

3.python >= 3.8
代码存在python3.8才有的操作












