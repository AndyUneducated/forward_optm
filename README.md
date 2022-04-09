# forward_optm

## 介绍

这份示例利用一个 forward function 和 backward function 做为基础，通过融合卷积神经网络、BatchNorm、Relu 来降低对内存的访问。

本文代码基于本人本科阶段写的一个数字手写字符识别的 demo 项目。由于 tf 1 和 tf 的 API 有了较大变化，因此环境成了一个问题。

## TensorFlow 中的 forward，backward 函数

TF 中的 forward 函数只会被 call 一次，这是 tf 与 pytorch 的一个显著不同。TF 是 graph based，所以 forward function 只会被 call 一次。TF 会自己寻找那个需要被计算的 node 来计算下一个 node，但不会再 call forward function。

详细来说， pytorch 中，定义连接网络的 forward，在运行时不必被调用。其原因是，当编写的网络继承 nn.Module 类时，forward 是在传入数据的时候，被 __call__ 方法调用的。具体步骤：

- 调用module的call方法
- module的call里面调用module的forward方法
- forward里面如果碰到Module的子类，回到第1步，如果碰到的是Function的子类，继续往下
- 调用Function的call方法
- Function的call方法调用了Function的forward方法。
- Function的forward返回值
- module的forward返回值
- 在module的call进行forward_hook操作，然后返回值。


本项目暂时以 tf 中的 forward，backward 实现为例。

以 exp 函数为例的 forward，backward 原型，仅作参考。具体实现请见 mnist_forward.py 和 mnist_backward.py：

```python
import tensorflow as tf

@tf.custom_gradient
def relu(x):
  y = tf.where(x>=0., x, 0.)

  def grad(dy):
    dx = tf.where(y>0., 1., 0.)
    return dy * dx

  return y, grad

@tf.custom_gradient
def exp(x):
  def grad(dy):
    dx = tf.math.exp(x)
    return dy * dx
  return tf.math.exp(x), grad
```

## BatchNorm

主要解决 Internal Covariant Shift 问题。深度学习对数据分布依赖较强，如果输入数据的分布情况差异大，则学习的开销、效果都不会乐观。

BN 步骤如下：

- 批量数据的 mean
- 批量数据的方差
- 批量数据的 normalize
- 加入输入的凭一变量，缩放变量
- 最后得出 normalize 的数值

代码原型：

详见 BatchNorm.py

BN 并不是适用于所有任务的，在 image-to-image 这样的任务中，尤其是超分辨率上，图像的绝对差异显得尤为重要，所以 batchnorm 的 scale 特性并不适合。

## tensorfolding

在 inference 阶段，BN layer 可以被预先计算 的 weight 捕捉，因此，BN layer 可以被代替。

原先需要 3-5 pass，但经过 Fused BN，只需要一个 pass。BN 中的 mean、var 可以被 merge 到 kernel。（只需要修改公式，见原论文）。

