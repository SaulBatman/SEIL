from e2cnn import gspaces
import e2cnn.nn as nn
import torch
import matplotlib.pyplot as plt

c4_act = gspaces.Rot2dOnR2(4)
layer1 = nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
                   nn.FieldType(c4_act, 1 * [c4_act.irrep(1)]),
                   3)

layer2 = torch.nn.Sequential(
    nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.regular_repr] + 1 * [c4_act.irrep(1)]),
              nn.FieldType(c4_act, 1 * [c4_act.irrep(2)]),
              1),
    # nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.regular_repr]),
    #           nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
    #           1)
)

a = torch.tensor([1, 2, 3, 4, -1, -1]).reshape(1, 6, 1, 1).float()
a_geo = nn.GeometricTensor(a, nn.FieldType(c4_act, 1* [c4_act.regular_repr] + 1 * [c4_act.irrep(1)]))
b = torch.tensor([4, 1, 2, 3, -1, -1]).reshape(1, 6, 1, 1).float()
b_geo = nn.GeometricTensor(b, nn.FieldType(c4_act, 1 * 1* [c4_act.regular_repr] + [c4_act.irrep(1)]))
# print(layer2(a_geo))
# print(layer2(b_geo))

layer3 = torch.nn.Sequential(
    nn.R2Conv(nn.FieldType(c4_act, 2 * [c4_act.regular_repr]),
              nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
              1),
)

a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).reshape(1, 8, 1, 1).float()
a_geo = nn.GeometricTensor(a, nn.FieldType(c4_act, 2* [c4_act.regular_repr]))
b = torch.tensor([4, 1, 2, 3, 5, 6, 7, 8]).reshape(1, 8, 1, 1).float()
b_geo = nn.GeometricTensor(b, nn.FieldType(c4_act, 2* [c4_act.regular_repr]))
print(layer3(a_geo))
print(layer3(b_geo))

layer3 = nn.R2Conv(nn.FieldType(c4_act, 1 * [c4_act.regular_repr] + 1 * [c4_act.irrep(1)]),
                   nn.FieldType(c4_act, 1 * [c4_act.trivial_repr]),
                   3)
a = torch.rand(1, 4, 3, 3)
b = torch.tensor([-1, -1]).reshape(1, 2, 1, 1).repeat(1, 1, 3, 3)
cat = torch.cat([a, b], 1)
cat_geo = nn.GeometricTensor(cat, nn.FieldType(c4_act, 1* [c4_act.regular_repr] + 1 * [c4_act.irrep(1)]))

# a = torch.zeros(1, 1, 3, 3)
# a[0, 0, 1, 0] = 1
# a_geo = nn.GeometricTensor(a, nn.FieldType(c4_act, [c4_act.trivial_repr]))
# out1 = layer1(a_geo)
#
# b = torch.zeros(1, 1, 3, 3)
# b[0, 0, 1, 2] = 1
# b_geo = nn.GeometricTensor(b, nn.FieldType(c4_act, [c4_act.trivial_repr]))
# out2 = layer1(b_geo)

a = torch.rand(1, 1, 3, 3)
a_geo = nn.GeometricTensor(a, nn.FieldType(c4_act, [c4_act.trivial_repr]))
for i in range(4):
    out = torch.tanh(layer1(a_geo.transform(i)).tensor.reshape(2).detach()).numpy()
    plt.scatter(out[1], out[0])
plt.show()


print(1)