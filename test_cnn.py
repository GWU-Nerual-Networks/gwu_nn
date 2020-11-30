from gwu_nn.cnn import CNN

cnn = CNN()
mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
mat3 = [[1]]
mat4 = [[0]]

assert cnn.weighted_sum(mat1, mat2) == 96
assert cnn.weighted_sum(mat3, mat4) == 0
assert cnn.flatten(mat1) == [1, 2, 3, 4, 5, 6, 7, 8, 9]

kernel1 = [[1, 1], [1, 1]]

assert cnn.conv_2d(mat1, kernel1) == [[12, 16], [24, 28]]

mat5 = [[1, 2, 3], [4, 5, 6]]

assert cnn.conv_2d(mat5, kernel1) == [[12, 16]]

mat6 = [[1, 2, 3, 4], [1, 2, 3, 4], [5, 6, 7, 8], [5, 6, 7, 8]]

assert cnn.max_pool(mat6, 2, 2) == [[2, 4], [6, 8]]
