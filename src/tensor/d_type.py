
# pytorch 中的所有数据类型都是 numbers， 并且 pytorch 会自动跟踪这些数据
# 通过 dtype 参数控制 默认 -- torch.float32
import torch
points_64 = torch.randn(5, dtype=torch.double)
print(points_64)