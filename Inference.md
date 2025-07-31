# 数据核对
```console
magik_compute.py
sess = tf.InteractiveSession()
b = tf.transpose(conv_res, [0, 3, 1, 2]) # nhwc2nchw
a = sess.run(b) # a can be print,type(a) : numpy.ndarray
m = 0
for h in range(a.shape[3]):
    if m<=3:
        print('===>',a[0,0,0,h])
        m+=1
exit()
```
```console
和venus核对前27，
feature==> (1, 226, 226, 3)
weight==> (3, 3, 3, 32)
    sess = tf.InteractiveSession()
    with tf.Session() as sess:
            feature,res = sess.run([feature,res])
    # feature,weight,res = sess.run([feature,weight,res])
    print('feature==>',feature[0,0:3,0:3,:])
    print('weight==>',weight[:,:,:,0])
    exit()
```

## Release error
Release模式不能使用MAGIK_INFERENCE_DUMP_PATH（使用DUMP导致参数更改），否则会爆以下错误：
```
Traceback (most recent call last):
  File "inference_magik_realtime.py", line 88, in <module>
    graph.build_graph(shape=inputs_shape)
  File "/data_bak/cywang/MagikInference/Magik/magik_graph.py", line 203, in build_graph
    op.run(self.output_blob)
  File "/data_bak/cywang/MagikInference/Magik/magik_ops_factor.py", line 163, in run
    self.op.run(input_blob)
  File "/data_bak/cywang/MagikInference/Magik/magik_ops.py", line 1040, in run
    super(BatchNormScaleInt8, self).run_end(blob, config)
  File "/data_bak/cywang/MagikInference/Magik/magik_ops.py", line 271, in run_end
    self.dump_feature_and_weight(blob)
  File "/data_bak/cywang/MagikInference/Magik/magik_ops.py", line 531, in dump_feature_and_weight
    feature = np.transpose(feature, (0, 3, 1, 2))
    ...
NotImplementedError: Cannot convert a symbolic Tensor (clip_by_value:0) to a numpy array.
```
## how to set shape for mutiple inputs 

### Release mod
从t40compare.pb看每个输入的名称和shape，按正序写成如下格式：
```
inputs_shape = {"input": (1, 2, 256, 4),
                    "onnx__QuantizeConcatInference_277":  (1, 1, 128, 32),
                    "onnx__QuantizeConcatInference_289":  (1, 1, 64, 32),
                    "hidden":  (64, 1, 32),
                    "onnx__QuantizeConcatInference_500":  (1, 1, 128, 32)
                }
```
### Debug mod
debug mod下，inputs_shape的key要写成0~n的格式（debug和release获取输入值时使用的key不同，release是读取输入的inputs的key，而debug是for循环，从0到n），同时，输入顺序需要调整，否则会报错，正确顺序可以通过graph.get_graph_input_name()获取
```
inputs_shape = {0: (1, 1, 128, 32),
                  1: (1, 1, 64, 32),
                  2: (1, 1, 64, 32),
                  3: (1, 1, 64, 32),
                  4: (1, 1, 64, 32),
                  5: (1, 1, 64, 32),
                  6: (1, 1, 64, 32),
                  7: (1, 1, 64, 32),
                  8: (1, 1, 64, 32),
                  9: (1, 1, 128, 32),
                  10: (1, 1, 256, 4),
                  11: (64, 1, 32),
    }
```

## DUMP数据量不对
magik_inference没有进行32对齐，从而导致和板端数据量对不齐
```console
## Magik/magik_ops.py
 def encode_ori_data(data_ori, BITWIDTH):
    # print(data_ori.shape,data_ori)
    # shape = data_ori.shape
    # remainder = shape[-1] % 32
    # if remainder != 0:
    #     padding = 32 - remainder
    # else:
    #     padding = 0

    # # 创建需要补充的数组
    # pad_shape = list(shape)
    # pad_shape[-1] = padding
    # pad_array = np.zeros(pad_shape, dtype=np.float32)

    # # 将补充数组拼接在原始数组后面
    # new_shape = list(shape)
    # new_shape[-1] += padding
    # data_ori = np.concatenate([data_ori, pad_array], axis=-1).reshape(new_shape)
```

## py3.6_torch1.8
2024-03-12 14:17:43.058789: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10.0'; dlerror: libcusparse.so.10.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/user/miniconda3/envs/py3.6_torch1.8/lib/python3.6/site-packages/cv2/../../lib64:
添加cuda10.0到环境变量，tensorflow1.15支持cuda10.0，默认的cuda10.2部分库无法起作用








