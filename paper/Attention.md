# Attention
原文名称：Attention Is All You Need
原文链接：https://arxiv.org/abs/1706.03762







###
为什么除以$\sqrt{d_k}$,
除以$\sqrt{d_k}$ 的原因在论文中的解释是“进行点乘后的数值很大，导致通过softmax后梯度变的很小”，所以通过除以$\sqrt{d_k}$ 来进行缩放。



#### 参考
[CSDN](https://blog.csdn.net/qq_37541097/article/details/117691873)