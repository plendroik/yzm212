import numpy as np

matris = np.array([
    [4, 1, 2, 3],
    [1, 2, 0, 1],
    [2, 0, 3, 1],
    [3, 1, 1, 4]
])

ozdegerler = np.linalg.eigvals(matris)

_, ozvektorler = np.linalg.eig(matris)

print("Özdeğerler: ", ozdegerler)
print("Özvektörler: ", ozvektorler)
