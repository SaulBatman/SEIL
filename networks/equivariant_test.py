from e2cnn import gspaces
import e2cnn.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

# c4_act = gspaces.Rot2dOnR2(4)
# layer1 = nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
#                    nn.FieldType(c4_act, 1 * [c4_act.irrep(1)]),
#                    3)
#
# a = torch.rand(1, 1, 3, 3)
# a_geo = nn.GeometricTensor(a, nn.FieldType(c4_act, [c4_act.trivial_repr]))
# for i in range(4):
#     out = torch.tanh(layer1(a_geo.transform(i)).tensor.reshape(2).detach()).numpy()
#     plt.scatter(out[1], out[0])
# plt.show()


c4_act = gspaces.Rot2dOnR2(4)

V = np.array([[1, 0],
              [0, 1],
              [1, 1],
              [-1, -1]])

V1 = np.array([[1, -1],
               [0, 1],
               [-1, 0],
               [-1, 1]])

W = []
W1 = []
for i in range(4):
    reg = c4_act.regular_repr.representation(i)
    std = c4_act.irrep(1).representation(i)
    W.append((reg @ V @ std).sum())
    W1.append((reg @ V1 @ std).sum())
print(1)