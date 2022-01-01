import numpy as np

z = b0 + b1 * X1 + b2 * X2
sigmoid = 1 / (1 + np.exp(-z))



if Y == 1:
    db0 = db0 + (1 - sigmoid)
    db1 = db1 + (1 - sigmoid) * X1
    db2 = db2 + (1 - sigmoid) * X2
else:
    db0 = db0 - sigmoid
    db1 = db1 - sigmoid * X1
    db2 = db2 - sigmoid * X2

b0 = b0 + alpha * db0
b1 = b1 + alpha * db1
b2 = b2 + alpha * db2