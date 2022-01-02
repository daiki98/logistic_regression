import numpy as np
import matplotlib.pyplot as plt


input_data = np.array([[1, 1, 0], [1, 4, 0], [3, 1, 0], [4, 5, 1], [0, 7, 1]])
data_number = input_data.shape[0]


epochs = 1000
alpha = 0.1

b0 = 0.1
b1 = 0.1
b2 = 0.1

for t in range(epochs):
    db0 = 0
    db1 = 0
    db2 = 0
    likelihood = 1
    
    for i in range(data_number):
        z = b0 + b1 * input_data[i, 0] + b2 * input_data[i, 1]
        sigmoid = 1 / (1 + np.exp(-z))

        if input_data[i, 2] == 1:
            db0 = db0 + (1 - sigmoid)
            db1 = db1 + (1 - sigmoid) * input_data[i, 0]
            db2 = db2 + (1 - sigmoid) * input_data[i, 1]
            
            likelihood = likelihood * (1 / (1 + np.exp(-z)))
        else:
            db0 = db0 - sigmoid
            db1 = db1 - sigmoid * input_data[i, 0]
            db2 = db2 - sigmoid * input_data[i, 1]
            likelihood = likelihood * (1 - (1 / (1 + np.exp(-z))))
        b0 = b0 + alpha * db0
        b1 = b1 + alpha * db1
        b2 = b2 + alpha * db2
        plt.scatter(t, likelihood)
x = np.linspace(0, 7, 100)
y = - b0/b2 - b1/b2 * x
# plt.plot(x, y)

# for n in range(data_number):
#     plt.scatter(input_data[n, 0], input_data[n, 1])
    
plt.show()
