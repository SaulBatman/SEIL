from e2cnn import gspaces
import e2cnn.nn as nn
import torch
import matplotlib.pyplot as plt

c4_act = gspaces.Rot2dOnR2(4)
layer1 = nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
                   nn.FieldType(c4_act, 1 * [c4_act.irrep(3)]),
                   3)

layer3 = nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
                   nn.FieldType(c4_act, 1 * [c4_act.regular_repr]),
                   3)

a = torch.zeros(1, 1, 3, 3)
a[0, 0, 1, 0] = 1
a_geo = nn.GeometricTensor(a, nn.FieldType(c4_act, [c4_act.trivial_repr]))
out1 = layer1(a_geo)

b = torch.zeros(1, 1, 3, 3)
b[0, 0, 1, 2] = 1
b_geo = nn.GeometricTensor(b, nn.FieldType(c4_act, [c4_act.trivial_repr]))
out2 = layer1(b_geo)


print(1)