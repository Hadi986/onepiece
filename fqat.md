# ops fx
ops torch版本无要求，最好>=1.6
fx torch>=1.8
# fqat数据出错
## 注：trainingkit和magikinference对数据，使用MAGIK_TRAININGKIT_DUMP_DEVP(1),dump出的数据为float
## 1>在对应的.cu中，找到要验证的位置，使用dump_data把数据dump出来，magikinference对应的数据解析保存（注意数据格式是nchw还是nhwc）
## 2>仅仅是某些数据出错，可以算出对应的index，在.cu根据index判断将其打印出来
```console
.cu
if (idx<=3){
    printf("=====>%d---%f",idx,res);
    }
```
```console
common.cc 
char *data_cpu_ptr = new char[size * sizeof(float)];
float* float_array = reinterpret_cast<float*>(data_cpu_ptr);//char2float,否则打印会乱码，注：此时数据还在gpu,要用printf打印
for (int i = 0; i < 5; i++){      
    printf("%f \n", float_array[i]);          
    }
```

## 4bit在ops_wrapper dump数据(如:unpool)
输出feature为4bit数据,要和dump的数据对齐，就得把相邻的两个数合并为一个8bit输出
```console
    input = self.fixpoint_forward(input)
    input_4bit = input.cpu().numpy()
    input_4bit = np.transpose(input_4bit, (0, 2, 3, 1))
    dump_data = np.zeros(input_4bit.shape[:-1] + (input_4bit.shape[-1] // 2,), dtype=np.uint8)
    for i in range(input_4bit.shape[-1] // 2):
        dump_data[:, :, :, i] = (input_4bit[:, :, :, 2*i + 1] * 16 + input_4bit[:, :, :, 2*i]).astype(np.uint8)
    d_input_path = "/data_bak/cywang/clients/yolov5s-plate/unpool.bin"
    dump_data.tofile(d_input_path)
```

# fqat网络修改常见错误：
### 1.output == n:
error
x1 = conv1(x1)
x2 = conv2(x2)
x1 = out_nop(x1)
x2 = out_nop(x2)
要在每一个卷积之后加out_nop
normal:
x1 = conv1(x1)
x1 = out_nop(x1)
x2 = conv2(x2)
x2 = out_nop(x2)
### 2.网络forward中不能出现list函数
eg:list(a,b)
修改===>x=[];x.append(a);x.append(b)
### 3.out_nop要添加在最后一层conv之后，且如果有多个输出，那么要在所有输出都添加了out_nop之后再进行其他操作
eg: for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        x[i] = self.out_nop(x[i]) if hasattr(self, "out_nop") else x[i]
==> output_list = []
    for i in range(self.nl):
        h1 = self.cv2[i](x[i])
        h1 = self.out_nop(h1)
        h2 = self.cv3[i](x[i])
        h2 = self.out_nop(h2)
        output_list.append([h1, h2])
    for i in range(self.nl):
        x[i] = torch.cat(output_list[i],1)
### 4.module can only call once
eg1: class Conv(nn.Module):
!!!        default_act = nn.SiLU()  # default activation
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            """Initialize Conv layer with given arguments including activation."""
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
==> class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            """Initialize Conv layer with given arguments including activation."""
            super().__init__()
            self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
            self.bn = nn.BatchNorm2d(c2)
!!!            self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
错误用法是因为在定义类时定义default_act为类属性，会导致其在trace的时候会被判别为进行了两次调用

eg2: class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d()
        def forward(self, x):
            y = self.conv(x);
            y = self.conv(y);
            return y
==> class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = nn.Conv2d()
            self.conv1 = nn.Conv2d()
        def forward(self, x):
            y = self.conv(x);
            y = self.conv1(y);
            return y

### 5.calibration data is None!
class PConv(nn.Module):
    def __init__(self,
                 g=True):
        super().__init__()
        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
    def forward_slicing(self, x):
    def forward_split_cat(self, x):
===>
class PConv(nn.Module):
    def __init__(self,
                 g=True):
        super().__init__()
        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
    def forward(self, x):
    def forward_split_cat(self, x):

# fqat量化位宽（板端）
feature为4，8，10，12
weight为4，8


# EMA(eval 精度一直不变化)
```console
for module in model.model.modules():
    try:
        module.set_run_fix_preprocess()
        print('1111111111')
        print('module',module.first_run_fix_preprocess)
    except:
        print('2222222222222')
```



