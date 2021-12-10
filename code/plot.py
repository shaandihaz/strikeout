from matplotlib import pyplot as plt

losses = [0.6920, 0.6865, 0.6881, 0.6767, 0.6774, 0.6882, 0.6926, 0.6700, 0.6674, 0.6714, 0.6527, 0.6754, 0.6704, 0.6946, 0.6680, 0.6904, 0.6760, 0.6234]
x = [i for i in range(len(losses))]
plt.plot(x, losses)
plt.title('Loss per batch')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.show()