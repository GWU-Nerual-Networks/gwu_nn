from gwu_nn.cnn import CNN

cnn = CNN()
mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
mat3 = [[1]]
mat4 = [[0]]

assert cnn.weighted_sum(mat1, mat2) == 96
assert cnn.weighted_sum(mat3, mat4) == 0
assert cnn.flatten(mat1) == [1, 2, 3, 4, 5, 6, 7, 8, 9]