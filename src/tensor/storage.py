import torch

# a pytorch tensor instance is a view of a storage instance
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])

print(points.storage())
print(points.storage()[0])

points.storage()[0] = 10
print(points)


# recognizable from a trailing underscore in their name
# like zero_,which indicates that the method operates in place
# by modifying the input instead of creating a new output tensor and returning it
a = torch.ones(3, 2)
a.zero_()
print(a)