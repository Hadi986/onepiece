# 量化

## 





### LLM Quant

- **AWQ**

  code:AutoAWQ

  [https://github.com/casper-hansen/AutoAWQ]: 

#### AWQ 量化

**核心观点 1：权重并不同等重要，仅有小部分显著权重对推理结果影响较大**

AWQ 量化发现，模型的权重并不同等重要，仅有 0.1%～1% 的小部分显著权重对模型输出精度影响较大。因此，如果能有办法只对 0.1%～1% 这一小部分权重保持原来的精度（如 FP16），对其他权重进行低比特量化，就可以在保持精度几乎不变的情况下，大幅降低模型内存占用，并提升推理速度。

**核心观点 2：量化时对显著权重进行放大可以降低量化误差**

在量化过程中，对显著权重进行放大可以显著减少量化误差。具体方法是通过引入激活感知缩放因子，对显著权重通道进行动态缩放，从而降低量化误差。

##### 挑选显著权重的方法

<img src="/home/user/awq.png" style="zoom:200%;" />

**基于激活值分布挑选**

- **方法**：通过分析激活值的分布，识别出需要保护的权重通道。激活值较大的通道对应的权重对模型输出影响显著，因此这些通道被视为显著权重通道。
- **步骤**：
  1. **计算激活值的绝对值平均值**：对激活值矩阵的每一列计算绝对值的平均值。
  2. **选择显著通道**：根据平均值的大小，选择前 0.1%～1% 的通道作为显著通道。
- **优势**：这种方法能够更准确地识别出对模型性能影响较大的权重通道，从而在量化过程中保留这些关键权重的精度。

**基于权重分布挑选**

- **方法**：对权重矩阵中的元素按绝对值大小由大到小排序，选择前 0.1%～1% 的元素作为显著权重。
- **步骤**：
  1. **排序权重**：对权重矩阵中的元素按绝对值大小进行排序。
  2. **选择显著权重**：选择排序后的前 0.1%～1% 的权重作为显著权重。
- **优势**：简单直接，但实验表明其效果不如基于激活值分布挑选显著权重的方法。

**对显著权重进行放大的策略**

**通道缩放优化**

- **方法**：引入激活感知缩放因子，对显著权重通道进行动态缩放。
- **步骤**：
  
  1. **确定缩放因子**：根据激活值分布确定缩放因子 \( s \)。
  2. **缩放显著权重**：将显著权重乘以缩放因子 \( s \)。
  3. **调整输入**：在计算时，将输入除以相同的缩放因子 \( s \)。
  4. **量化过程**：对缩放后的权重进行量化。
- **数学推导**：
  - 量化函数为：
    $$
    Q(w) = \text{Round}\left( \frac{w}{\Delta} \right)
    $$
    
    
  - 对于缩放后的权重 
    $$
     w \cdot s （ s > 1 ）
    $$
    ，量化误差变为：
    $$
    \text{Error} = \left\| \frac{w \cdot s}{\Delta} - Q\left( \frac{w \cdot s}{\Delta} \right) \right\|
    $$
    
    
  - 由于 \( s > 1 \)，权重被放大，量化误差相对减小。

均匀量化策略

- **方法**：采用均匀量化策略，避免混合精度带来的硬件适配问题。
- **步骤**：
  1. **量化所有权重**：对所有权重进行低比特量化。
  2. **缩放显著权重**：在量化过程中，对显著权重乘以较大的缩放因子 \( s \)，以降低量化误差。
  3. **调整非显著权重**：对非显著权重乘以较小的缩放因子 \( s \)，以减少计算资源的浪费。
- **优势**：硬件友好，支持单指令多数据（SIMD）优化，提升边缘设备计算效率。

##### 总结
AWQ 量化通过分析激活值分布来识别显著权重通道，并采用通道缩放优化策略对显著权重进行放大，从而显著减少量化误差。这种方法在保持模型性能的同时，大幅降低了模型的内存占用和计算量，适用于各种硬件平台。

- **GPTQ**

  # GPTQ 量化核心原理与细节介绍

  ## 一、核心原理

  GPTQ（Generalized Projection-based Quantization）是一种针对大语言模型（LLM）的训练后量化（PTQ）方法，旨在减少量化过程中的精度损失。其核心思想是通过利用二阶梯度信息来补偿量化误差，从而提高量化模型的性能。

  ## 二、实现细节

  ### （一）量化误差补偿

    * **二阶梯度计算** ：GPTQ 通过计算权重的二阶梯度来确定量化误差的补偿值。具体来说，对于每个权重参数，计算其对损失函数的二阶导数，这反映了权重变化对模型性能的敏感程度。

    * **误差补偿公式** ：量化后的权重表示为：
      $$
      \mathbf{w}_q = \text{Round}\left(\frac{\mathbf{w}}{\Delta}\right) \cdot \Delta + \mathbf{e}
      $$
      其中，\(Delta\) 是量化步长，\(\mathbf{e}\) 是误差补偿项，通过二阶梯度信息计算得到。

  ### （二）校准数据利用

    * **校准数据选择** ：GPTQ 需要使用少量校准数据（如 Pile 数据集）来估计权重的二阶梯度。这些数据应具有代表性，能够覆盖模型在实际应用中可能遇到的各种输入情况。
    * **校准过程** ：在校准过程中，GPTQ 通过前向传播和反向传播计算每个权重的二阶梯度，并根据这些信息确定误差补偿值。

  ### （三）硬件效率优化

    * **权重重排序** ：GPTQ 通过重排序权重来提高硬件计算效率。具体来说，将权重按照其对模型性能的影响程度进行排序，使得在量化过程中，更重要的权重能够得到更精确的表示。
    * **高效内核实现** ：GPTQ 支持高效的 CUDA 内核实现，通过权重打包和核融合优化推理速度。例如，采用 4 位权重量化时，通过特定的打包技术将多个权重存储在一个 32 位或 16 位的寄存器中，从而减少内存访问次数。

  ### （四）适用场景与模型支持

    * **通用模型量化** ：GPTQ 适用于各种通用语言模型，如 LLaMA、OPT 等。它能够在保持模型性能的同时，显著减少模型的存储和计算需求。
    * **快速量化需求** ：由于 GPTQ 的量化过程相对较快，因此适用于需要快速量化的场景，如模型部署前的快速优化。

  ## 三、优势总结

    * **精度保持** ：通过二阶梯度信息补偿量化误差，GPTQ 能够在低比特量化下保持较高的模型精度。
    * **硬件友好** ：权重重排序和高效内核实现使得 GPTQ 在各种硬件平台上都能高效运行。
    * **广泛适用** ：适用于多种语言模型和应用场景，具有良好的通用性和扩展性。

  ## 四、实际应用示例

  ```python
  from gptq import GPTQ
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  model_name = "llama-2-7b"
  quantized_model_name = "llama-2-7b-gptq-4bit"
  calibration_data = "pile_sample.json"
  
  # 加载模型和校准数据
  model = AutoModelForCausalLM.from_pretrained(model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  # 初始化 GPTQ 量化器
  gptq = GPTQ(model, bits=4)
  
  # 执行量化
  gptq.quantize(calibration_data)
  
  # 保存量化后的模型
  gptq.save_quantized_model(quantized_model_name)
  ```

  此代码将 Llama-2-7B 模型量化为 4 位，量化后的模型在保持较高性能的同时，显著减少了存储和计算需求。

- 开源支持	AutoAWQ、TinyChat 框架	GPTQ - for - LLaMa、ExLlama 优化

- RTN
