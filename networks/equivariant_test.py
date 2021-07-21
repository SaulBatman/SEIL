from e2cnn import gspaces
import e2cnn.nn as nn
import torch
import matplotlib.pyplot as plt

d4_act = gspaces.FlipRot2dOnR2(4)
layer1 = nn.R2Conv(nn.FieldType(d4_act, 1*[d4_act.trivial_repr]),
                  nn.FieldType(d4_act, 1*[d4_act.quotient_repr((None, 4))]),
                  3)
layer2 = nn.R2Conv(nn.FieldType(d4_act, 1*[d4_act.trivial_repr]),
                  nn.FieldType(d4_act, 1*[d4_act.quotient_repr((0, 4))]),
                  3)
layer3 = nn.R2Conv(nn.FieldType(d4_act, 1*[d4_act.trivial_repr]),
                  nn.FieldType(d4_act, 1*[d4_act.regular_repr]),
                  3)

a = torch.zeros(1, 1, 3, 3)
a[0, 0, 1, 0] = 1
a_geo = nn.GeometricTensor(a, nn.FieldType(d4_act, [d4_act.trivial_repr]))
out1 = layer1(a_geo)

b = torch.zeros(1, 1, 3, 3)
b[0, 0, 1, 2] = 1
b_geo = nn.GeometricTensor(b, nn.FieldType(d4_act, [d4_act.trivial_repr]))
out2 = layer1(b_geo)


r_act = gspaces.Flip2dOnR2()
layer3 = nn.R2Conv(nn.FieldType(r_act, 1*[r_act.trivial_repr]),
                  nn.FieldType(r_act, 1*[r_act.regular_repr]),
                  3)
a_geo = nn.GeometricTensor(a, nn.FieldType(r_act, [r_act.trivial_repr]))
b_geo = nn.GeometricTensor(b, nn.FieldType(r_act, [r_act.trivial_repr]))
out1 = layer3(a_geo)
out2 = layer3(b_geo)


print(1)