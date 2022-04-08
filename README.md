# forward_optm

## 介绍

这份示例利用一个 forward function 和 backward function 做为基础，通过融合卷积神经网络、BatchNorm、Relu 来降低对内存的访问。

## BatchNorm

BN 并不是适用于所有任务的，在 image-to-image 这样的任务中，尤其是超分辨率上，图像的绝对差异显得尤为重要，所以 batchnorm 的 scale 并不适合。